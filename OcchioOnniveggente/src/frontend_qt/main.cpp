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

    // When using qt_add_qml_module CMake helper, QML files are embedded under
    // the prefix `qt/qml/<URI>/`.  The previous URL was missing this prefix,
    // causing the application to fail to locate the QML resource at runtime.
    // See: https://doc.qt.io/qt-6/qtqml-cppintegration-topic.html

    const QUrl url(u"qrc:/qt/qml/Oracolo/MainWindow.qml"_s);
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreationFailed,
                     &app, [](){ QCoreApplication::exit(-1); }, Qt::QueuedConnection);
    engine.load(url);

    return app.exec();
}
