#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include "RealtimeClient.h"
using namespace Qt::StringLiterals;

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    RealtimeClient client;
    engine.rootContext()->setContextProperty("realtimeClient", &client);

    // Load the main QML file from the Oracolo module.
    engine.loadFromModule(u"Oracolo"_s, u"MainWindow"_s);

    if (engine.rootObjects().isEmpty())
        return -1;
    return app.exec();
}
