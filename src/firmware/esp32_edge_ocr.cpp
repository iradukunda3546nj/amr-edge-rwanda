/**
 * AMR-Edge Rwanda: edge node firmware (reference implementation, ESP-IDF v5.x)
 * ---------------------------------------------------------------------------
 * Wake on RTC timer -> capture grayscale frame -> crop the installer-calibrated
 * wheel ROIs -> INT8 digit inference (digit_model.cpp, ESP-NN) -> validate and
 * detect continuous night flow -> seal a 20-byte payload (AES-128-CTR +
 * AES-CMAC-32) -> MQTT 3.1.1 PUBLISH QoS 1 over TLS through a SIM7600 LTE
 * Cat-1 modem -> power the modem off -> deep sleep.
 *
 * Hardware: ESP32-S3 (8 MB PSRAM), OV2640, SIMCom SIM7600 (LTE Cat-1, 3G
 * fallback), LiSOCl2 cell + hybrid layer capacitor for the modem's TX peaks.
 * Components: espressif/esp32-camera, espressif/esp-nn.
 *
 * The payload layout and the MQTT topic match src/web/js/core/telemetry.js
 * and the dashboard simulation exactly.
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sys/time.h>

#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "mbedtls/aes.h"
#include "mbedtls/cipher.h"
#include "mbedtls/cmac.h"
#include "nvs.h"
#include "nvs_flash.h"

#include "digit_model.h"
#include "digit_model_int8.h"

static const char *TAG = "amr-edge";

// ============================================================================
// Board (Seeed XIAO ESP32-S3 Sense camera pinout + carrier board)
// ============================================================================
namespace pins {
constexpr int CAM_XCLK = 10, CAM_SIOD = 40, CAM_SIOC = 39;
constexpr int CAM_D7 = 48, CAM_D6 = 11, CAM_D5 = 12, CAM_D4 = 14, CAM_D3 = 16, CAM_D2 = 18, CAM_D1 = 17, CAM_D0 = 15;
constexpr int CAM_VSYNC = 38, CAM_HREF = 47, CAM_PCLK = 13;
constexpr gpio_num_t CAM_PWR_EN = GPIO_NUM_1;    // camera rail load switch
constexpr gpio_num_t LED_EN = GPIO_NUM_2;        // illumination ring
constexpr gpio_num_t MODEM_PWRKEY = GPIO_NUM_3;  // SIM7600 PWRKEY (via NPN, active high here)
constexpr gpio_num_t MODEM_PWR_EN = GPIO_NUM_4;  // modem VBAT load switch: zero sleep current
constexpr gpio_num_t VBAT_SENSE_EN = GPIO_NUM_5; // switched battery divider
constexpr int MODEM_TX = 43, MODEM_RX = 44;
}  // namespace pins

// ============================================================================
// Configuration. Identity, key and ROIs are provisioned into NVS at install.
// ============================================================================
namespace cfg {
constexpr char APN[] = "internet.mtn.rw";
constexpr char BROKER[] = "mqtts://telemetry.wasac.example:8883";
constexpr int TZ_OFFSET_S = 2 * 3600;  // CAT (UTC+2)
constexpr int SLOT_NIGHT_1 = 2 * 3600; // two night reads -> minimum night flow
constexpr int SLOT_NIGHT_2 = 4 * 3600;
constexpr int SLOT_REPORT = 6 * 3600;  // daily report
constexpr float MIN_CONFIDENCE = 0.50f;
constexpr float LEAK_MNF_LPH = 15.0f;  // continuous flow threshold at night
constexpr int MAX_WHEELS = 10;
}  // namespace cfg

struct Roi {
    uint16_t x, y, w, h;
    uint8_t fractional;  // red drum: sub-m3 register
};

struct Provisioning {
    uint32_t meter_id;
    uint8_t key[16];
    uint8_t n_wheels;
    Roi wheels[cfg::MAX_WHEELS];
};

static bool load_provisioning(Provisioning &p) {
    nvs_handle_t h;
    if (nvs_open("amr", NVS_READONLY, &h) != ESP_OK) return false;
    size_t key_len = sizeof p.key, roi_len = sizeof p.wheels;
    bool ok = nvs_get_u32(h, "meter_id", &p.meter_id) == ESP_OK &&
              nvs_get_blob(h, "key", p.key, &key_len) == ESP_OK && key_len == 16 &&
              nvs_get_u8(h, "n_wheels", &p.n_wheels) == ESP_OK && p.n_wheels <= cfg::MAX_WHEELS &&
              nvs_get_blob(h, "rois", p.wheels, &roi_len) == ESP_OK;
    nvs_close(h);
    return ok;
}

// ============================================================================
// State kept across deep sleep (RTC slow memory)
// ============================================================================
struct RtcState {
    uint32_t magic;
    uint32_t last_epoch;   // last transmitted epoch (strictly increasing: replay protection)
    float night1_m3;
    uint32_t night1_epoch;
    bool time_valid;
};
RTC_DATA_ATTR static RtcState g_rtc;
constexpr uint32_t RTC_MAGIC = 0xA3E0D6E2;

// ============================================================================
// 20-byte payload (see telemetry.js for the field table)
// ============================================================================
#pragma pack(push, 1)
struct Payload {
    uint8_t hdr;        // version(3b) | flags(5b)
    uint32_t meter_id;
    uint32_t epoch;
    uint32_t reading_dl;
    int16_t flow_lph;   // 0x7FFF = not measured
    uint8_t battery;    // (mV - 2000) / 10
    uint8_t tag[4];
};
#pragma pack(pop)
static_assert(sizeof(Payload) == 20, "payload must be 20 bytes");

enum : uint8_t { FLAG_LEAK = 0x01, FLAG_ENHANCED = 0x02, FLAG_LOW_CONF = 0x04, FLAG_RATE_FAULT = 0x08, FLAG_LOW_BATT = 0x10 };
constexpr uint8_t PROTO_VERSION = 1;
constexpr int16_t FLOW_UNAVAILABLE = 0x7FFF;

static void seal(Payload &p, const uint8_t key[16]) {
    uint8_t *raw = reinterpret_cast<uint8_t *>(&p);
    uint8_t counter[16] = {};
    memcpy(counter, raw + 1, 8);  // meter_id || epoch || 0^8
    uint8_t stream[16];
    size_t off = 0;
    mbedtls_aes_context aes;
    mbedtls_aes_init(&aes);
    mbedtls_aes_setkey_enc(&aes, key, 128);
    mbedtls_aes_crypt_ctr(&aes, 7, &off, counter, stream, raw + 9, raw + 9);
    mbedtls_aes_free(&aes);
    uint8_t mac[16];
    mbedtls_cipher_cmac(mbedtls_cipher_info_from_type(MBEDTLS_CIPHER_AES_128_ECB), key, 128, raw, 16, mac);  // CONFIG_MBEDTLS_CMAC_C
    memcpy(p.tag, mac, 4);
}

// ============================================================================
// Camera
// ============================================================================
static bool camera_on() {
    gpio_set_direction(pins::CAM_PWR_EN, GPIO_MODE_OUTPUT);
    gpio_set_level(pins::CAM_PWR_EN, 1);
    vTaskDelay(pdMS_TO_TICKS(10));
    camera_config_t c = {};
    c.pin_pwdn = -1;
    c.pin_reset = -1;
    c.pin_xclk = pins::CAM_XCLK;
    c.pin_sccb_sda = pins::CAM_SIOD;
    c.pin_sccb_scl = pins::CAM_SIOC;
    c.pin_d7 = pins::CAM_D7; c.pin_d6 = pins::CAM_D6; c.pin_d5 = pins::CAM_D5; c.pin_d4 = pins::CAM_D4;
    c.pin_d3 = pins::CAM_D3; c.pin_d2 = pins::CAM_D2; c.pin_d1 = pins::CAM_D1; c.pin_d0 = pins::CAM_D0;
    c.pin_vsync = pins::CAM_VSYNC;
    c.pin_href = pins::CAM_HREF;
    c.pin_pclk = pins::CAM_PCLK;
    c.xclk_freq_hz = 20000000;
    c.ledc_timer = LEDC_TIMER_0;
    c.ledc_channel = LEDC_CHANNEL_0;
    c.pixel_format = PIXFORMAT_GRAYSCALE;  // luma only, as the model expects
    c.frame_size = FRAMESIZE_VGA;
    c.fb_count = 1;
    c.fb_location = CAMERA_FB_IN_PSRAM;
    c.grab_mode = CAMERA_GRAB_LATEST;
    return esp_camera_init(&c) == ESP_OK;
}

static void camera_off() {
    esp_camera_deinit();
    gpio_set_level(pins::CAM_PWR_EN, 0);
}

static camera_fb_t *capture() {
    gpio_set_direction(pins::LED_EN, GPIO_MODE_OUTPUT);
    gpio_set_level(pins::LED_EN, 1);
    for (int i = 0; i < 3; ++i) {  // let auto-exposure settle under the LED
        if (camera_fb_t *warm = esp_camera_fb_get()) esp_camera_fb_return(warm);
    }
    camera_fb_t *fb = esp_camera_fb_get();
    gpio_set_level(pins::LED_EN, 0);
    return fb;
}

// ============================================================================
// Reading
// ============================================================================
struct Reading {
    double m3;
    float min_conf;
    bool readable;
};

static Reading read_register(const camera_fb_t *fb, const Provisioning &prov) {
    static uint8_t crop[256 * 256];
    Reading r = {0, 1.0f, true};
    double integer = 0, fraction = 0, frac_scale = 1;
    for (int i = 0; i < prov.n_wheels; ++i) {
        const Roi &roi = prov.wheels[i];
        const int w = roi.w > 256 ? 256 : roi.w, h = roi.h > 256 ? 256 : roi.h;
        for (int y = 0; y < h; ++y) memcpy(crop + y * w, fb->buf + (roi.y + y) * fb->width + roi.x, w);
        const DigitResult d = digit_model_classify(crop, w, h);
        r.min_conf = fminf(r.min_conf, d.confidence);
        if (d.cls == DIGIT_REJECT_CLASS) {
            r.readable = false;
            continue;
        }
        if (roi.fractional) {
            frac_scale /= 10;
            fraction += d.cls * frac_scale;
        } else {
            integer = integer * 10 + d.cls;
        }
    }
    r.m3 = integer + fraction;
    return r;
}

// ============================================================================
// SIM7600 modem: AT command driver with the built-in MQTT(S) client
// ============================================================================
class Modem {
   public:
    bool power_on() {
        uart_config_t u = {};
        u.baud_rate = 115200;
        u.data_bits = UART_DATA_8_BITS;
        u.parity = UART_PARITY_DISABLE;
        u.stop_bits = UART_STOP_BITS_1;
        u.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
        u.source_clk = UART_SCLK_DEFAULT;
        uart_driver_install(port_, 2048, 0, 0, nullptr, 0);
        uart_param_config(port_, &u);
        uart_set_pin(port_, pins::MODEM_TX, pins::MODEM_RX, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
        gpio_set_direction(pins::MODEM_PWR_EN, GPIO_MODE_OUTPUT);
        gpio_set_direction(pins::MODEM_PWRKEY, GPIO_MODE_OUTPUT);
        gpio_set_level(pins::MODEM_PWR_EN, 1);
        vTaskDelay(pdMS_TO_TICKS(100));
        gpio_set_level(pins::MODEM_PWRKEY, 1);  // SIM7600: PWRKEY low >= 500 ms to power on
        vTaskDelay(pdMS_TO_TICKS(600));
        gpio_set_level(pins::MODEM_PWRKEY, 0);
        for (int i = 0; i < 30; ++i) {
            if (at("AT", "OK", 500)) return at("ATE0", "OK", 500) && at("AT+CPIN?", "+CPIN: READY", 5000);
            vTaskDelay(pdMS_TO_TICKS(500));
        }
        return false;
    }

    bool attach(const char *apn, int timeout_s = 90) {
        char cmd[96];
        snprintf(cmd, sizeof cmd, "AT+CGDCONT=1,\"IP\",\"%s\"", apn);
        if (!at(cmd, "OK", 2000)) return false;
        at("AT+CNMP=2", "OK", 2000);  // automatic: LTE preferred, 3G fallback
        const int64_t deadline = esp_timer_get_time() + int64_t(timeout_s) * 1000000;
        while (esp_timer_get_time() < deadline) {
            // +CEREG / +CGREG stat 1 = home, 5 = roaming
            if (at("AT+CEREG?", "+CEREG: 0,1", 1000) || contains("+CEREG: 0,5") ||
                at("AT+CGREG?", "+CGREG: 0,1", 1000) || contains("+CGREG: 0,5"))
                return true;
            vTaskDelay(pdMS_TO_TICKS(2000));
        }
        return false;
    }

    /** NITZ network time -> UTC epoch, 0 on failure. +CCLK: "yy/MM/dd,hh:mm:ss+zz" (zz in quarter hours). */
    uint32_t network_time() {
        if (!at("AT+CCLK?", "+CCLK:", 1000)) return 0;
        int yy, MM, dd, hh, mm, ss, q = 0;
        char sign = '+';
        const char *s = strchr(rx_, '"');
        if (!s || sscanf(s + 1, "%d/%d/%d,%d:%d:%d%c%d", &yy, &MM, &dd, &hh, &mm, &ss, &sign, &q) < 6) return 0;
        if (yy < 24) return 0;  // modem reports its default clock until NITZ arrives
        struct tm t = {};
        t.tm_year = yy + 100; t.tm_mon = MM - 1; t.tm_mday = dd; t.tm_hour = hh; t.tm_min = mm; t.tm_sec = ss;
        setenv("TZ", "UTC0", 1);
        tzset();
        return uint32_t(mktime(&t) - (sign == '-' ? -1 : 1) * q * 15 * 60);
    }

    /** MQTT 3.1.1 over TLS (client certificate pre-loaded with AT+CCERTDOWN at provisioning). */
    bool publish(const char *client_id, const char *topic, const uint8_t *payload, size_t len) {
        char cmd[160];
        at("AT+CSSLCFG=\"sslversion\",0,4", "OK", 1000);  // TLS 1.2
        at("AT+CSSLCFG=\"authmode\",0,2", "OK", 1000);    // mutual authentication
        at("AT+CSSLCFG=\"cacert\",0,\"wasac_ca.pem\"", "OK", 1000);
        at("AT+CSSLCFG=\"clientcert\",0,\"device.pem\"", "OK", 1000);
        at("AT+CSSLCFG=\"clientkey\",0,\"device.key\"", "OK", 1000);
        if (!at("AT+CMQTTSTART", "+CMQTTSTART: 0", 12000)) return false;
        snprintf(cmd, sizeof cmd, "AT+CMQTTACCQ=0,\"%s\",1", client_id);
        bool ok = at(cmd, "OK", 2000) && at("AT+CMQTTSSLCFG=0,0", "OK", 2000);
        snprintf(cmd, sizeof cmd, "AT+CMQTTCONNECT=0,\"%s\",60,1", cfg::BROKER);
        ok = ok && at(cmd, "+CMQTTCONNECT: 0,0", 30000);
        ok = ok && send_block("AT+CMQTTTOPIC=0,", reinterpret_cast<const uint8_t *>(topic), strlen(topic));
        ok = ok && send_block("AT+CMQTTPAYLOAD=0,", payload, len);
        ok = ok && at("AT+CMQTTPUB=0,1,60", "+CMQTTPUB: 0,0", 30000);  // QoS 1: returns after PUBACK
        at("AT+CMQTTDISC=0,60", "+CMQTTDISC: 0,0", 10000);
        at("AT+CMQTTREL=0", "OK", 2000);
        at("AT+CMQTTSTOP", "+CMQTTSTOP: 0", 10000);
        return ok;
    }

    void power_off() {
        at("AT+CPOF", "OK", 5000);
        vTaskDelay(pdMS_TO_TICKS(500));
        gpio_set_level(pins::MODEM_PWR_EN, 0);
        uart_driver_delete(port_);
    }

   private:
    static constexpr uart_port_t port_ = UART_NUM_1;
    char rx_[512] = {};

    bool at(const char *cmd, const char *expect, int timeout_ms) {
        uart_flush_input(port_);
        uart_write_bytes(port_, cmd, strlen(cmd));
        uart_write_bytes(port_, "\r\n", 2);
        return wait(expect, timeout_ms);
    }

    bool send_block(const char *prefix, const uint8_t *data, size_t len) {
        char cmd[48];
        snprintf(cmd, sizeof cmd, "%s%u", prefix, unsigned(len));
        if (!at(cmd, ">", 2000)) return false;
        uart_write_bytes(port_, data, len);
        return wait("OK", 3000);
    }

    bool wait(const char *expect, int timeout_ms) {
        size_t n = 0;
        rx_[0] = '\0';
        const int64_t deadline = esp_timer_get_time() + int64_t(timeout_ms) * 1000;
        while (esp_timer_get_time() < deadline && n < sizeof rx_ - 1) {
            const int r = uart_read_bytes(port_, reinterpret_cast<uint8_t *>(rx_ + n), sizeof rx_ - 1 - n, pdMS_TO_TICKS(20));
            if (r <= 0) continue;
            n += r;
            rx_[n] = '\0';
            if (strstr(rx_, expect)) return true;
            if (strstr(rx_, "ERROR")) return false;
        }
        return false;
    }

    bool contains(const char *s) const { return strstr(rx_, s) != nullptr; }
};

// ============================================================================
// Power and scheduling
// ============================================================================
static uint16_t battery_mv() {
    gpio_set_direction(pins::VBAT_SENSE_EN, GPIO_MODE_OUTPUT);
    gpio_set_level(pins::VBAT_SENSE_EN, 1);
    adc_oneshot_unit_handle_t adc;
    adc_oneshot_unit_init_cfg_t u = {};
    u.unit_id = ADC_UNIT_1;
    adc_oneshot_new_unit(&u, &adc);
    adc_oneshot_chan_cfg_t c = {};
    c.atten = ADC_ATTEN_DB_12;
    c.bitwidth = ADC_BITWIDTH_12;
    adc_oneshot_config_channel(adc, ADC_CHANNEL_6, &c);
    int raw = 0, acc = 0;
    for (int i = 0; i < 8; ++i) { adc_oneshot_read(adc, ADC_CHANNEL_6, &raw); acc += raw; }
    adc_oneshot_del_unit(adc);
    gpio_set_level(pins::VBAT_SENSE_EN, 0);
    return uint16_t((acc / 8) * 3100 / 4095 * 2);  // 1:2 divider; calibrate per unit
}

enum class Slot { Night1, Night2, Report, Unscheduled };

static int local_second(uint32_t epoch) { return int((epoch + cfg::TZ_OFFSET_S) % 86400); }

static Slot slot_of(uint32_t epoch) {
    const int s = local_second(epoch);
    auto near = [s](int t) { return abs(s - t) < 600; };
    if (near(cfg::SLOT_NIGHT_1)) return Slot::Night1;
    if (near(cfg::SLOT_NIGHT_2)) return Slot::Night2;
    if (near(cfg::SLOT_REPORT)) return Slot::Report;
    return Slot::Unscheduled;
}

[[noreturn]] static void sleep_until_next_slot() {
    timeval tv;
    gettimeofday(&tv, nullptr);
    int64_t wait = 3600;  // without valid time, retry hourly
    if (g_rtc.time_valid) {
        const int now = local_second(uint32_t(tv.tv_sec));
        wait = 86400;
        for (int t : {cfg::SLOT_NIGHT_1, cfg::SLOT_NIGHT_2, cfg::SLOT_REPORT}) {
            int d = t - now;
            if (d <= 60) d += 86400;
            if (d < wait) wait = d;
        }
    }
    gpio_hold_en(pins::CAM_PWR_EN);
    gpio_hold_en(pins::MODEM_PWR_EN);
    gpio_deep_sleep_hold_en();
    esp_sleep_enable_timer_wakeup(uint64_t(wait) * 1000000ULL);
    esp_deep_sleep_start();
}

// ============================================================================
// Main cycle
// ============================================================================
extern "C" void app_main() {
    gpio_deep_sleep_hold_dis();
    gpio_hold_dis(pins::CAM_PWR_EN);
    gpio_hold_dis(pins::MODEM_PWR_EN);
    if (g_rtc.magic != RTC_MAGIC) g_rtc = {RTC_MAGIC, 0, 0, 0, false};

    nvs_flash_init();
    Provisioning prov;
    if (!load_provisioning(prov) || !digit_model_init()) {
        ESP_LOGE(TAG, "not provisioned or model init failed");
        sleep_until_next_slot();
    }

    timeval tv;
    gettimeofday(&tv, nullptr);
    uint32_t epoch = uint32_t(tv.tv_sec);
    const Slot slot = g_rtc.time_valid ? slot_of(epoch) : Slot::Unscheduled;

    // 1-3. Capture and read
    const int64_t t0 = esp_timer_get_time();
    Reading reading = {0, 0, false};
    if (camera_on()) {
        if (camera_fb_t *fb = capture()) {
            reading = read_register(fb, prov);
            esp_camera_fb_return(fb);
        }
        camera_off();
    }
    ESP_LOGI(TAG, "reading %.4f m3, min conf %.3f, %lld ms", reading.m3, reading.min_conf, (esp_timer_get_time() - t0) / 1000);

    // 4. Validation and minimum-night-flow leak detection
    uint8_t flags = 0;
    const bool accepted = reading.readable && reading.min_conf >= cfg::MIN_CONFIDENCE;
    if (!accepted) flags |= FLAG_LOW_CONF;
    int16_t mnf = FLOW_UNAVAILABLE;
    bool transmit = slot == Slot::Report || slot == Slot::Unscheduled;
    if (accepted && slot == Slot::Night1) {
        g_rtc.night1_m3 = float(reading.m3);
        g_rtc.night1_epoch = epoch;
    } else if (accepted && slot == Slot::Night2 && g_rtc.night1_epoch) {
        const float hours = float(epoch - g_rtc.night1_epoch) / 3600.0f;
        const float lph = (float(reading.m3) - g_rtc.night1_m3) * 1000.0f / hours;
        mnf = int16_t(fminf(fmaxf(lph, 0.0f), 32766.0f));
        if (lph > cfg::LEAK_MNF_LPH) {
            flags |= FLAG_LEAK;
            transmit = true;  // alert now rather than at the 06:00 report
        }
    }
    const uint16_t batt = battery_mv();
    if (batt < 3300) flags |= FLAG_LOW_BATT;

    // 5-6. Seal and publish
    if (transmit || !g_rtc.time_valid) {
        Modem modem;
        if (modem.power_on() && modem.attach(cfg::APN)) {
            if (const uint32_t net = modem.network_time()) {
                timeval set = {time_t(net), 0};
                settimeofday(&set, nullptr);
                epoch = net;
                g_rtc.time_valid = true;
            }
            if (transmit && g_rtc.time_valid && epoch > g_rtc.last_epoch) {
                Payload p = {};
                p.hdr = uint8_t((PROTO_VERSION << 5) | (flags & 0x1F));
                p.meter_id = prov.meter_id;
                p.epoch = epoch;
                p.reading_dl = accepted && reading.m3 < 429496.0 ? uint32_t(llround(reading.m3 * 10000.0)) : 0;
                p.flow_lph = mnf;
                p.battery = uint8_t(batt > 2000 ? (batt - 2000) / 10 : 0);
                seal(p, prov.key);
                char client_id[24], topic[48];
                snprintf(client_id, sizeof client_id, "MTR-%lu", static_cast<unsigned long>(prov.meter_id));
                snprintf(topic, sizeof topic, "wasac/v1/meters/%lu/up", static_cast<unsigned long>(prov.meter_id));
                if (modem.publish(client_id, topic, reinterpret_cast<const uint8_t *>(&p), sizeof p)) g_rtc.last_epoch = epoch;
                else ESP_LOGW(TAG, "publish failed; retried at the next slot");
            }
        }
        modem.power_off();
        if (slot == Slot::Report) g_rtc.night1_epoch = 0;
    }

    sleep_until_next_slot();
}
