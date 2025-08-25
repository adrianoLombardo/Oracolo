import QtQuick
import QtQuick.Effects

Rectangle {
  id: root
  property alias contentItem: content
  color: "#10182A"; radius: 18; border.color: "#1B263B"; border.width: 1

  layer.enabled: true
  layer.effect: MultiEffect {
    shadowEnabled: true
    shadowColor: "#00E5FF"
    shadowBlur: 0.55
    shadowHorizontalOffset: 0
    shadowVerticalOffset: 0
    brightness: 0.03
  }

  default property alias data: content.data
  Item { id: content; anchors.fill: parent; anchors.margins: 16 }
}
