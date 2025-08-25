import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Oracolo
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
      Label { text: "Parziali: " + rt.partial; color: "#D7FFF9"; wrapMode: Text.Wrap }
    }

    ListView {
      id: chatView
      Layout.fillWidth: true; Layout.fillHeight: true
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
