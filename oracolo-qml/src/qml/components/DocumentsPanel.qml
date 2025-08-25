import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../Palette.js" as Palette

Rectangle {
  id: panel
  color: Palette.card
  radius: 18
  border.color: Palette.accent
  border.width: 1
  property alias contentItem: content

  ColumnLayout {
    id: content
    anchors.fill: parent
    anchors.margins: 16
    spacing: 8
  }
}
