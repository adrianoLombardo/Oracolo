#include "RealtimeClient.h"
#include <QAudioDevice>
#include <QMediaDevices>
#include <QJsonDocument>
#include <QJsonObject>
#include <QtEndian>
#include <cmath>

RealtimeClient::RealtimeClient(QObject* parent) : QObject(parent) {
    connect(&m_ws, &QWebSocket::connected, this, &RealtimeClient::onConnected);
    connect(&m_ws, &QWebSocket::disconnected, this, &RealtimeClient::onDisconnected);
    connect(&m_ws, &QWebSocket::textMessageReceived, this, &RealtimeClient::onTextMessage);
    connect(&m_ws, &QWebSocket::binaryMessageReceived, this, &RealtimeClient::onBinaryMessage);

    // Audio format: 24kHz, mono, PCM 16-bit
    m_format.setSampleRate(24000);
    m_format.setChannelCount(1);
#if QT_VERSION >= QT_VERSION_CHECK(6, 2, 0)
    m_format.setSampleFormat(QAudioFormat::Int16);
#else
    m_format.setSampleSize(16);
    m_format.setCodec("audio/pcm");
    m_format.setByteOrder(QAudioFormat::LittleEndian);
    m_format.setSampleType(QAudioFormat::SignedInt);
#endif
}

void RealtimeClient::setupAudio() {
    QAudioDevice dev = QMediaDevices::defaultAudioOutput();
    m_sink = std::make_unique<QAudioSink>(dev, m_format, this);
    m_output = m_sink->start(); // push mode: we write() PCM bytes here
}

void RealtimeClient::connectToUrl(const QUrl& url) {
    if (m_connected) return;
    m_ws.open(url);
}

void RealtimeClient::disconnectFromServer() {
    if (!m_connected) return;
    m_ws.close();
}

void RealtimeClient::onConnected() {
    m_connected = true;
    emit connectedChanged();

    if (!m_sink) setupAudio();

    // handshake
    QJsonObject hello{
        {"type", "hello"},
        {"sr", 24000},
        {"format", "pcm_s16le"},
        {"channels", 1}
    };
    m_ws.sendTextMessage(QString::fromUtf8(QJsonDocument(hello).toJson(QJsonDocument::Compact)));
}

void RealtimeClient::onDisconnected() {
    m_connected = false;
    emit connectedChanged();
}

void RealtimeClient::onTextMessage(const QString& text) {
    const auto doc = QJsonDocument::fromJson(text.toUtf8());
    if (!doc.isObject()) return;
    const auto obj = doc.object();
    const auto type = obj.value("type").toString();
    if (type == "partial") {
        m_partial = obj.value("text").toString();
        emit partialChanged();
    } else if (type == "answer") {
        m_answer = obj.value("text").toString();
        emit answerChanged();
    } // else: ignore
}

void RealtimeClient::onBinaryMessage(const QByteArray& data) {
    // stream to audio device
    if (m_output) {
        m_output->write(data);
    }
    // compute simple RMS for VU
    const int16_t* s = reinterpret_cast<const int16_t*>(data.constData());
    const int count = data.size() / 2;
    if (count > 0) {
        double sumsq = 0.0;
        for (int i=0; i<count; ++i) {
            const double v = s[i] / 32768.0;
            sumsq += v*v;
        }
        double rms = std::sqrt(sumsq / count);
        double newLevel = std::clamp(rms * 2.0, 0.0, 1.0); // scale a gusto
        if (std::abs(newLevel - m_level) > 0.01) {
            m_level = newLevel;
            emit levelChanged();
        }
    }
}

void RealtimeClient::sendBargeIn() {
    if (!m_connected) return;
    QJsonObject obj{{"type", "barge_in"}};
    m_ws.sendTextMessage(QString::fromUtf8(QJsonDocument(obj).toJson(QJsonDocument::Compact)));
}
