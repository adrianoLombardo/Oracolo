import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
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
                        NeonIconButton {
                            anchors.horizontalCenter: parent.horizontalCenter
                            iconText: "\uD83C\uDFA4" // microphone emoji
                            onClicked: Bridge.onMicTapped()
                        }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "WAKE WORD RECOGNIZED"
                            font.family: Theme.font
                            font.pixelSize: 12
                            color: Bridge.wakeWordRecognized ? Theme.neonA : Theme.textSoft
                        }
                    }
                }

                NeonPanel {
                    id: rightPanel
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom
                    width: parent.width * 0.55

                    Column {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 12

                        Flickable {
                            id: chatView
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.bottom: inputRow.top
                            contentWidth: width
                            clip: true
                            Column {
                                id: chatColumn
                                width: chatView.width
                                spacing: 10
                                ChatBubble { text: "Quali sono gli orari di apertura?"; fromUser: true }
                                ChatBubble { text: "Certamente. Gli orari di apertura del museo sono dal martedì alla domenica, dalle 9:00 alle 19:00, il lunedì siamo chiusi."; fromUser: false }
                            }
                        }

                        Row {
                            id: inputRow
                            width: parent.width
                            spacing: 8
                            TextField {
                                id: inputField
                                width: parent.width - sendButton.width - 8
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
                                id: sendButton
                                width: 56
                                height: 56
                                iconText: "\u27A4"
                                onClicked: Bridge.onSendPressed()
                            }
                        }
                    }
                }
            }
        }
    }
}
