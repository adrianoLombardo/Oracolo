import QtQuick
import QtQuick.Controls
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
}
