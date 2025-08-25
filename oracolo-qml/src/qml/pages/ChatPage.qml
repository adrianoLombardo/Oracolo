import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Oracolo
import "../Palette.js" as Palette
import "../components"

Item {
  anchors.fill: parent

  ListModel { id: conversationModel }

  Connections {
    target: rt
    function onAnswerChanged() {
      if (rt.answer.length) {
        conversationModel.append({ text: rt.answer, fromUser: false })
      }
    }
  }

  ColumnLayout {
    anchors.fill: parent; spacing: 12
    anchors.bottomMargin: inputRow.height + 12

    NeonCard {
      Layout.fillWidth: true; Layout.preferredHeight: 120
      color: Palette.card
      Label { text: "Parziali: " + rt.partial; color: Palette.text; wrapMode: Text.Wrap }
    }

    ListView {
      id: chatView
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

      model: conversationModel
      delegate: ChatBubble { text: model.text; fromUser: model.fromUser }

    }
  }

  Row {
    id: inputRow
    anchors.bottom: parent.bottom
    anchors.right: parent.right
    anchors.margins: 12
    spacing: 8

    TextField {
      id: input
      width: 240
      placeholderText: "Scrivi un prompt testuale (facoltativo)…"
    }

    RoundButton {
      text: "\u{1F3A4}"
      onClicked: rt.sendBargeIn()
    }
  }
}
