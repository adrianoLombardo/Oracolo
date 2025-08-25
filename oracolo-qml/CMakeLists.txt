cmake_minimum_required(VERSION 3.20)
project(OracoloUI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS
    Quick Qml QuickControls2 WebSockets Multimedia)

qt_standard_project_setup()

qt_add_executable(OracoloUI
    src/main.cpp
    src/RealtimeClient.cpp
    src/RealtimeClient.h
)

qt_add_qml_module(OracoloUI
    URI Oracolo
    VERSION 1.0
    QML_FILES
        src/qml/Main.qml
        src/qml/pages/ChatPage.qml
        src/qml/pages/DocumentsPage.qml
        src/qml/pages/SettingsPage.qml
        src/qml/components/NeonCard.qml
        src/qml/components/VUMeter.qml
        src/qml/components/Waveform.qml
)

target_link_libraries(OracoloUI PRIVATE
    Qt6::Quick
    Qt6::Qml
    Qt6::QuickControls2
    Qt6::WebSockets
    Qt6::Multimedia
)
