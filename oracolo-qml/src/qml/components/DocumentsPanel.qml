import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Column {
  id: root
  width: parent ? parent.width : 200
  property color textColor: "#D7FFF9"
  property var documents: [
    { name: "mostra.md" },
    { name: "orari.pdf" }
  ]

  signal documentSelected(string name)

  spacing: 8

  Label {
    text: "Documenti"
    color: root.textColor
    font.pixelSize: 16
    font.bold: true
  }

  ListView {
    id: list
    width: parent.width
    height: contentHeight
    model: root.documents

    delegate: Item {
      width: list.width
      height: 28

      Row {
        anchors.verticalCenter: parent.verticalCenter
        spacing: 8
        Label {
          text: iconFor(model.name)
          font.pixelSize: 16
        }
        Label {
          text: model.name
          color: root.textColor
        }
      }

      MouseArea {
        anchors.fill: parent
        onClicked: root.documentSelected(model.name)
      }
    }
  }

  function iconFor(name) {
    var ext = name.split('.').pop().toLowerCase()
    if (ext === 'md') return "\uD83D\uDCDD"; // memo
    if (ext === 'pdf') return "\uD83D\uDCC4"; // page
    return "\uD83D\uDCC4";
  }
}
