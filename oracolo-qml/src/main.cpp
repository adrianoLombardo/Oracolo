#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QtQuickControls2>
#include "RealtimeClient.h"

int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);
    QQuickStyle::setStyle("Fusion"); // base, poi customizzi in QML

    QQmlApplicationEngine engine;

    // Istanza client realtime (WS + audio)
    auto *rt = new RealtimeClient(&engine);
    engine.rootContext()->setContextProperty("rt", rt);

    const QUrl url(u"qrc:/Oracolo/src/qml/Main.qml"_qs);
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated, &app,
                     [url](QObject *obj, const QUrl &objUrl) {
                         if (!obj && url == objUrl) QCoreApplication::exit(-1);
                     }, Qt::QueuedConnection);
    engine.load(url);

    return app.exec();
}
