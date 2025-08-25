import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Qt5Compat.GraphicalEffects

Rectangle {
    id: root
    width: 1024
    height: 640
    color: "#0a0d12"
    border.color: "#3df5ff"
    border.width: 2

    // Neon glow effect: attempt to use MultiEffect from QtQuick.Effects (Qt 6.5+).
    // If QtQuick.Effects is missing (older Qt versions), fall back to
    // DropShadow from Qt5Compat.GraphicalEffects.
    layer.enabled: true
    layer.effect: fallbackShadow

    // Fallback DropShadow effect; replaced with MultiEffect if available
    Component {
        id: fallbackShadow
        DropShadow {
            color: "#3df5ff"
            samples: 32
            horizontalOffset: 0
            verticalOffset: 0
        }
    }

    // Attempt to dynamically load MultiEffect. Failure leaves DropShadow active.
    Component.onCompleted: {
        try {
            var component = Qt.createComponent("import QtQuick.Effects; MultiEffect { shadowEnabled: true; shadowColor: '#3df5ff' }");
            if (component.status === Component.Ready) {
                root.layer.effect = component;
            }
        } catch (e) {
            // QtQuick.Effects module not present; using DropShadow as fallback
        }
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

