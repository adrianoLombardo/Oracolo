import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects

Window {
    id: root
    visible: true
    width: 1024
    height: 640
    color: "#0a0d12"
    property var docs: []
    property var rules: []

    Rectangle {
        anchors.fill: parent
        color: "transparent"
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
                Button {
                    text: "Docs"
                    onClicked: root.docs = backend.get_documents()
                }
                Button {
                    text: "Save"
                    onClicked: backend.save_config()
                }
            }

            RowLayout {
                spacing: 12

                TextField {
                    id: ruleField
                    placeholderText: "kw1, kw2"
                    Layout.fillWidth: true
                }
                Button {
                    text: "Update"
                    onClicked: root.rules = backend.update_rules(ruleField.text)
                }
            }

            ListView {
                id: docList
                model: root.docs
                Layout.fillWidth: true
                Layout.fillHeight: true
                delegate: Text {
                    text: modelData.title ? modelData.title : modelData
                    color: "#3df5ff"
                }
            }

            Text {
                text: root.rules.join(", ")
                color: "#3df5ff"
                wrapMode: Text.Wrap
                Layout.fillWidth: true
            }
        }
    }
}

