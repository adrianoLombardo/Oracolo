#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include "RealtimeClient.h"

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    RealtimeClient client;
    engine.rootContext()->setContextProperty("realtimeClient", &client);

    const QUrl url(u"qrc:/Oracolo/MainWindow.qml"_qs);
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreationFailed,
                     &app, [](){ QCoreApplication::exit(-1); }, Qt::QueuedConnection);
    engine.load(url);

    return app.exec();
}
