#pragma once
#include <QObject>
#include <QWebSocket>
#include <QAudioFormat>
#include <QAudioSink>
#include <QIODevice>
#include <QTimer>
#include <memory>

class RealtimeClient : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool connected READ connected NOTIFY connectedChanged)
    Q_PROPERTY(QString partial READ partial NOTIFY partialChanged)
    Q_PROPERTY(QString answer READ answer NOTIFY answerChanged)
    Q_PROPERTY(qreal level READ level NOTIFY levelChanged)

public:
    explicit RealtimeClient(QObject* parent = nullptr);

    Q_INVOKABLE void connectToUrl(const QUrl& url);
    Q_INVOKABLE void disconnectFromServer();
    Q_INVOKABLE void sendBargeIn();

    bool connected() const { return m_connected; }
    QString partial() const { return m_partial; }
    QString answer() const { return m_answer; }
    qreal level() const { return m_level; }

signals:
    void connectedChanged();
    void partialChanged();
    void answerChanged();
    void levelChanged();
    void error(QString message);

private slots:
    void onConnected();
    void onTextMessage(const QString& text);
    void onBinaryMessage(const QByteArray& data);
    void onDisconnected();

private:
    void setupAudio();

    QWebSocket m_ws;
    bool m_connected = false;

    QString m_partial;
    QString m_answer;
    qreal m_level = 0.0; // 0..1

    QAudioFormat m_format;
    std::unique_ptr<QAudioSink> m_sink;
    QIODevice* m_output = nullptr; // device returned by sink->start()
};
