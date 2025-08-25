cmake_minimum_required(VERSION 3.16)
project(oracolo_client LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 6.2 COMPONENTS Quick Qml WebSockets Multimedia REQUIRED)
qt_policy(SET QTP0001 NEW)

qt_add_executable(oracolo_client
    main.cpp
    RealtimeClient.cpp
    RealtimeClient.h
)

target_include_directories(oracolo_client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

qt_add_qml_module(oracolo_client
    URI Oracolo
    VERSION 1.0
    QML_FILES
        qml/MainWindow.qml
        qml/DocumentTable.qml
        qml/RulesPanel.qml
)

target_link_libraries(oracolo_client
    PRIVATE Qt6::Quick Qt6::WebSockets Qt6::Multimedia)

install(TARGETS oracolo_client
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
