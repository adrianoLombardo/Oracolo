import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Oracolo
import "../Palette.js" as Palette

Item {
  anchors.fill: parent

  ColumnLayout {
    anchors.fill: parent; spacing: 12

    NeonCard {
      Layout.fillWidth: true; Layout.preferredHeight: 120
      color: Palette.card
      Label { text: "Parziali: " + rt.partial; color: Palette.text; wrapMode: Text.Wrap }
    }

    NeonCard {
      Layout.fillWidth: true; Layout.fillHeight: true
      color: Palette.card
      Column {
        anchors.fill: parent; anchors.margins: 0; spacing: 6
        Label { text: "Ultima risposta"; color: Palette.accentLight; font.bold: true }
        Flickable {
          anchors.left: parent.left; anchors.right: parent.right; anchors.bottom: parent.bottom; anchors.top: previous.bottom
          contentWidth: parent.width; contentHeight: ansText.paintedHeight
          clip: true
          Text {
            id: ansText; width: parent.width; color: Palette.text; wrapMode: Text.Wrap
            text: rt.answer.length ? rt.answer : "—"
          }
        }
      }
    }

    RowLayout {
      Layout.fillWidth: true
      TextField {
        id: input; Layout.fillWidth: true; placeholderText: "Scrivi un prompt testuale (facoltativo)…"
      }
      Button {
        text: "Barge-in"
        onClicked: rt.sendBargeIn()
      }
    }
  }
}
