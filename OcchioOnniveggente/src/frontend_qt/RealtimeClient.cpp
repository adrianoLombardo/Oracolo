#include "RealtimeClient.h"
#include <QJsonDocument>
#include <QAudioFormat>
#include <QMediaDevices>
#include <QAudioSink>

RealtimeClient::RealtimeClient(QObject *parent) : QObject(parent)
{
    connect(&m_socket, &QWebSocket::connected, this, &RealtimeClient::onConnected);
    connect(&m_socket, &QWebSocket::binaryMessageReceived,
            this, &RealtimeClient::onBinaryMessageReceived);
    connect(&m_socket, &QWebSocket::textMessageReceived,
            this, &RealtimeClient::onTextMessageReceived);
}

void RealtimeClient::connectToServer(const QUrl &url)
{
    m_socket.open(url);
}

void RealtimeClient::sendCommand(const QString &type, const QVariantMap &payload)
{
    QJsonObject obj = QJsonObject::fromVariantMap(payload);
    obj.insert("type", type);
    m_socket.sendTextMessage(QJsonDocument(obj).toJson(QJsonDocument::Compact));
}

void RealtimeClient::sendHello(int sampleRate, int channels)
{
    QVariantMap payload{{"sr", sampleRate}, {"format", "pcm16"}, {"channels", channels}};
    sendCommand("hello", payload);
}

void RealtimeClient::sendText(const QString &text)
{
    QVariantMap payload{{"text", text}};
    sendCommand("message", payload);
}

void RealtimeClient::requestDocuments()
{
    QJsonObject obj{{"type", "list_docs"}};
    m_socket.sendTextMessage(QJsonDocument(obj).toJson(QJsonDocument::Compact));
}

void RealtimeClient::applyRules(const QJsonObject &rules)
{
    QJsonObject obj{{"type", "apply_rules"}, {"rules", rules}};
    m_socket.sendTextMessage(QJsonDocument(obj).toJson(QJsonDocument::Compact));
}

void RealtimeClient::onConnected()
{
    // placeholder for post-connection logic
}

void RealtimeClient::onBinaryMessageReceived(const QByteArray &message)
{
    if (!m_audioOutput) {
        QAudioFormat format;
        format.setSampleRate(24000);
        format.setChannelCount(1);
        format.setSampleFormat(QAudioFormat::Int16);
        m_audioOutput = new QAudioSink(QMediaDevices::defaultAudioOutput(), format, this);
        m_audioDevice = m_audioOutput->start();
    }
    if (m_audioDevice)
        m_audioDevice->write(message);
}

void RealtimeClient::onTextMessageReceived(const QString &message)
{
    QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());
    if (!doc.isObject())
        return;

    QJsonObject obj = doc.object();
    const QString type = obj.value("type").toString();
    if (type == "doc_list") {
        emit documentsReceived(obj.value("docs").toArray());
    } else if (type == "rule_update") {
        emit ruleUpdated(obj.value("rule").toObject());
    } else if (type == "policy_status") {
        emit policyStatusReceived(obj.value("status").toObject());
    } else {
        emit jsonMessageReceived(obj);
    }
}
