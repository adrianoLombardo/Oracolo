import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Effects 1.15

Rectangle {
    id: root
    width: 1024
    height: 640
    color: "#0a0d12"
    border.color: "#3df5ff"
    border.width: 2

    // neon glow effect
    layer.enabled: true
    layer.effect: DropShadow {
        color: "#3df5ff"
        radius: 20
        samples: 25
        verticalOffset: 0
        horizontalOffset: 0
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 24

        Text {
            text: "Occhio Onniveggente"
            color: "#3df5ff"
            font.pixelSize: 32
            font.family: "Consolas"
        }

        Slider {
            id: volumeSlider
            from: 0
            to: 100
            value: 50
            Layout.fillWidth: true
        }

        Switch {
            id: glowSwitch
            text: "Glow"
            checked: true
        }

        RowLayout {
            spacing: 12

            Button {
                text: "Start"
                onClicked: backend.start()
            }

            Button {
                text: "Reload"
                onClicked: backend.reload()
            }
        }
    }
}

