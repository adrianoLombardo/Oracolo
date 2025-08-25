#include "RealtimeClient.h"
#include <QJsonDocument>
#include <QAudioFormat>

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

void RealtimeClient::sendHello(int sampleRate, int channels)
{
    QJsonObject obj{{"type", "hello"}, {"sr", sampleRate}, {"format", "pcm16"}, {"channels", channels}};
    m_socket.sendTextMessage(QJsonDocument(obj).toJson(QJsonDocument::Compact));
}

void RealtimeClient::sendText(const QString &text)
{
    QJsonObject obj{{"type", "message"}, {"text", text}};
    m_socket.sendTextMessage(QJsonDocument(obj).toJson(QJsonDocument::Compact));
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
        m_audioOutput = new QAudioOutput(format, this);
        m_audioDevice = m_audioOutput->start();
    }
    if (m_audioDevice)
        m_audioDevice->write(message);
}

void RealtimeClient::onTextMessageReceived(const QString &message)
{
    QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());
    if (doc.isObject()) {
        QJsonObject obj = doc.object();
        if (obj.contains("documents") && obj.value("documents").isArray())
            emit documentsReceived(obj.value("documents").toArray());
        emit jsonMessageReceived(obj);
    }
}
