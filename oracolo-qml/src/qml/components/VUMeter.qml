import QtQuick
import "../Palette.js" as Palette

Item {
  id: root
  property real level: 0.0 // 0..1
  width: 200; height: 16
  Rectangle {
    anchors.fill: parent; radius: 8; color: Qt.darker(Palette.bg, 1.2)
    Rectangle {
      anchors.verticalCenter: parent.verticalCenter
      height: parent.height
      width: parent.width * root.level
      radius: 8
      gradient: Gradient {
        GradientStop { position: 0.0; color: Palette.accent }
        GradientStop { position: 1.0; color: Palette.accentLight }
      }
    }
  }
}
