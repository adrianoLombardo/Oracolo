import QtQuick
import "../Palette.js" as Palette

Item {
  id: orb
  property real radius: 100
  width: radius * 2
  height: width
  opacity: 0.2

  Rectangle {
    anchors.fill: parent
    radius: width / 2
    gradient: Gradient {
      GradientStop { position: 0.0; color: Palette.accent }
      GradientStop { position: 1.0; color: "transparent" }
    }
  }
}
