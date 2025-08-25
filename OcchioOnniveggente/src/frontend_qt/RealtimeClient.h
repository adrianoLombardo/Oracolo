#pragma once

#include <QObject>
#include <QWebSocket>
#include <QAudioOutput>
#include <QIODevice>
#include <QJsonObject>
#include <QUrl>

class RealtimeClient : public QObject
{
    Q_OBJECT
public:
    explicit RealtimeClient(QObject *parent = nullptr);

    Q_INVOKABLE void connectToServer(const QUrl &url);
    Q_INVOKABLE void sendHello(int sampleRate, int channels);
    Q_INVOKABLE void sendText(const QString &text);

signals:
    void jsonMessageReceived(const QJsonObject &obj);

private slots:
    void onConnected();
    void onBinaryMessageReceived(const QByteArray &message);
    void onTextMessageReceived(const QString &message);

private:
    QWebSocket m_socket;
    QAudioOutput *m_audioOutput = nullptr;
    QIODevice *m_audioDevice = nullptr;
};
