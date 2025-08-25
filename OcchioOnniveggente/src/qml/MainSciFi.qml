import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects

Rectangle {
    id: root
    width: 1024
    height: 640
    color: "#0a0d12"
    border.color: "#3df5ff"
    border.width: 2

    layer.enabled: true
    layer.effect: MultiEffect {
        shadowEnabled: true
        shadowColor: "#3df5ff"
        shadowBlur: 1.0
        shadowHorizontalOffset: 0
        shadowVerticalOffset: 0
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

