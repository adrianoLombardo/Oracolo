import QtQuick 2.15
import QtQuick.Controls 2.15
import "../Palette.js" as Palette

Rectangle {
  id: bubble
  property alias text: label.text
  property bool fromUser: false
  color: fromUser ? Palette.accent : Palette.card
  radius: 12
  border.color: Palette.accent
  border.width: fromUser ? 0 : 1
  anchors.margins: 4
  implicitWidth: label.implicitWidth + 24
  implicitHeight: label.implicitHeight + 16

  Label {
    id: label
    anchors.fill: parent
    anchors.margins: 8
    wrapMode: Text.Wrap
    color: fromUser ? Palette.bg : Palette.text
  }

Item {
    id: root
    property alias text: message.text
    property bool fromUser: false

    width: ListView.view ? ListView.view.width : 360
    implicitHeight: bubble.height + 8

    Rectangle {
        id: bubble
        radius: 8
        color: root.fromUser ? "#66FFF2" : "#333333"
        anchors.left: root.fromUser ? undefined : parent.left
        anchors.right: root.fromUser ? parent.right : undefined
        anchors.margins: 8
        width: Math.min(message.paintedWidth + 24, root.width * 0.8)
        height: message.paintedHeight + 16

        Text {
            id: message
            anchors.fill: parent
            anchors.margins: 8
            wrapMode: Text.Wrap
            color: root.fromUser ? "#000000" : "#D7FFF9"
        }
    }
}
