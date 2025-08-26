import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "."
import Theme 1.0
import "components"

Window {
    id: root
    width: 1100
    height: 700
    visible: true
    color: Theme.bg
    title: "Occhio Onniveggente"

    NeonPanel {
        id: frame
        anchors.fill: parent
        anchors.margins: 24

        Column {
            anchors.fill: parent
            anchors.margins: 24
            spacing: 24

            Row {
                id: topBar
                spacing: 30
                anchors.horizontalCenter: parent.horizontalCenter
                NeonTabButton { text: "DOCUMENTI" }
                NeonTabButton { text: "IMPOSTAZIONI" }
                NeonTabButton { text: "STRUMENTI" }
                NeonTabButton { text: "SERVER" }
                NeonTabButton { text: "LOG" }
            }

            Row {
                id: mainRow
                spacing: 24
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: topBar.bottom
                anchors.bottom: parent.bottom

                NeonPanel {
                    id: leftPanel
                    width: parent.width * 0.45
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom

                    Column {
                        anchors.fill: parent
                        anchors.margins: 24
                        spacing: 24
                        anchors.verticalCenter: parent.verticalCenter
                        NeonOrb {
                            anchors.horizontalCenter: parent.horizontalCenter
                            width: 220
                            height: 220
                        }
                        Waveform {
                            anchors.horizontalCenter: parent.horizontalCenter
                            width: 260
                            height: 60
                        }
                    }
                }

                NeonPanel {
                    id: rightPanel
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom
                    width: parent.width * 0.55

                    Item {
                        anchors.fill: parent
                        anchors.margins: 16

                        ChatHistory {
                            id: chatView
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.bottom: inputRow.top
                            historyModel: Bridge.historyModel
                        }

                        Row {
                            id: inputRow
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.bottom: logArea.top
                            spacing: 8
                            TextField {
                                id: inputField
                                anchors.verticalCenter: parent.verticalCenter
                                width: parent.width - sendButton.width - micButton.width - 16
                                color: Theme.text
                                placeholderText: "Scrivi..."
                                placeholderTextColor: Theme.textSoft
                                font.family: Theme.font
                                background: Rectangle {
                                    color: Theme.panel
                                    radius: Theme.radius
                                    border.color: Theme.border
                                    border.width: Theme.borderW
                                }
                            }
                            NeonIconButton {
                                id: micButton
                                width: 56
                                height: 56
                                iconText: "\uD83C\uDFA4"
                                onClicked: Bridge.onMicTapped()
                            }
                            NeonIconButton {
                                id: sendButton
                                width: 56
                                height: 56
                                iconText: "\u27A4"
                                onClicked: { Bridge.sendText(inputField.text); inputField.text = "" }
                            }
                        }

                        TextArea {
                            id: logArea
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.bottom: parent.bottom
                            height: 120
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            text: Bridge.logText
                            color: Theme.text
                            font.family: Theme.font
                            background: Rectangle {
                                color: Theme.panel
                                radius: Theme.radius
                                border.color: Theme.border
                                border.width: Theme.borderW
                            }
                        }
                    }
                }
            }
        }
    }
}
