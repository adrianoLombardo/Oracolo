import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Effects 1.15
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

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24

        TabBar {
            id: tabs
            Layout.fillWidth: true
            TabButton { text: "Chat" }
            TabButton { text: "Documenti" }
            TabButton { text: "Impostazioni" }
        }

        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: tabs.currentIndex

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

            ColumnLayout {
                anchors.fill: parent
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    TextField {
                        id: searchField
                        Layout.fillWidth: true
                        placeholderText: "Cerca..."
                        onTextChanged: docTable.filterText = text
                    }
                    Button {
                        text: "Aggiorna"
                        onClicked: realtimeClient.requestDocuments()
                    }
                }

                RulesPanel {
                    id: rulesPanel
                    Layout.fillWidth: true
                    onApplyRules: realtimeClient.applyRules(rules)
                }

                DocumentTable {
                    id: docTable
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    onSelectedDocumentChanged: previewArea.text = selectedDocument ? JSON.stringify(selectedDocument, null, 2) : ""
                }

                TextArea {
                    id: previewArea
                    readOnly: true
                    Layout.fillWidth: true
                    Layout.preferredHeight: 120
                    text: "Seleziona un documento"
                }

                Component.onCompleted: realtimeClient.requestDocuments()

                Connections {
                    target: realtimeClient
                    function onDocumentsReceived(docs) {
                        docTable.documents = docs
                    }
                }
            }

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
        function onRuleUpdated(rule) {
            ruleLabel.text = "Regola: " + (rule.description || rule.name || JSON.stringify(rule))
        }
        function onPolicyStatusReceived(status) {
            policyLabel.text = "Policy: " + (status.state || status.status || JSON.stringify(status))
        }
    }
}
