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

    ListModel {
        id: docModel
    }

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
            ColumnLayout {
                anchors.fill: parent
                spacing: 8

                TableView {
                    id: docTable
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: docModel
                    clip: true
                    delegate: Rectangle {
                        implicitHeight: 32
                        width: docTable.width
                        color: index % 2 ? "#151a1f" : "#1e242b"
                        Text {
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.left: parent.left
                            anchors.leftMargin: 8
                            text: name
                            color: "white"
                        }
                    }
                }
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
                Label { id: ruleLabel; text: "Regola: -" }
                Label { id: policyLabel; text: "Policy: -" }
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
        function onDocListReceived(docs) {
            docModel.clear()
            for (var i = 0; i < docs.length; ++i) {
                var item = docs[i]
                if (typeof item === "string")
                    docModel.append({ name: item })
                else if (item.name)
                    docModel.append({ name: item.name })
                else if (item.title)
                    docModel.append({ name: item.title })
                else
                    docModel.append({ name: JSON.stringify(item) })
            }
        }
        function onRuleUpdated(rule) {
            ruleLabel.text = "Regola: " + (rule.description || rule.name || JSON.stringify(rule))
        }
        function onPolicyStatusReceived(status) {
            policyLabel.text = "Policy: " + (status.state || status.status || JSON.stringify(status))
        }
    }
}
