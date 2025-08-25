import QtQuick
import QtQuick.Effects
import "../Palette.js" as Palette

Rectangle {
  id: root
  property alias contentItem: content
  color: Palette.card; radius: 18; border.color: Qt.darker(Palette.card, 1.3); border.width: 1

  layer.enabled: true
  layer.effect: MultiEffect {
    shadowEnabled: true
    shadowColor: Palette.accent
    shadowBlur: 0.55
    shadowHorizontalOffset: 0
    shadowVerticalOffset: 0
    brightness: 0.03
  }

  default property alias data: content.data
  Item { id: content; anchors.fill: parent; anchors.margins: 16 }
}
