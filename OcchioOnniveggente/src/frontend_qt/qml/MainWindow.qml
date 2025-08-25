import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects

ApplicationWindow {
    id: root
    width: 1024
    height: 640
    visible: true
    color: "#0a0d12"
    title: "Occhio Onniveggente"

    ComboBox {
        id: modeBox
        model: ["Museo", "Galleria", "Conferenze", "Didattica"]
        anchors.left: parent.left
        anchors.leftMargin: 16
        anchors.top: parent.top
        anchors.topMargin: 16
        onCurrentTextChanged: {
            realtimeClient.sendText(JSON.stringify({type: "mode", value: currentText}))
        }
    }

    TabView {
        id: tabs
        anchors.fill: parent
        anchors.margins: 24

        Tab {
            title: "Chat"
            ColumnLayout {
                anchors.fill: parent
                spacing: 12

                TextArea {
                    id: chatArea
                    readOnly: true
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    textFormat: TextEdit.PlainText
                }

                RowLayout {
                    Layout.fillWidth: true
                    TextField {
                        id: input
                        Layout.fillWidth: true
                        placeholderText: "Scrivi..."
                        onAccepted: {
                            chatArea.text += "Tu: " + text + "\n"
                            realtimeClient.sendText(text)
                            text = ""
                        }
                    }
                    Button {
                        text: "Invia"
                        onClicked: input.accepted()
                    }
                }
            }
        }

        Tab {
            title: "Documenti"
            Label {
                anchors.centerIn: parent
                text: "Elenco documenti..."
            }
        }

        Tab {
            title: "Impostazioni"
            Column {
                anchors.margins: 12
                anchors.fill: parent
                spacing: 8
                Label { text: "Volume generale" }
                Slider { from: 0; to: 1; value: 1 }
                CheckBox { text: "Modalità realtime"; checked: true }
            }
        }
    }

    Connections {
        target: realtimeClient
        function onJsonMessageReceived(obj) {
            if (obj.text) {
                chatArea.text += "Oracolo: " + obj.text + "\n"
            }
        }
    }
}
