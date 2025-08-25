import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
  anchors.fill: parent

  GridLayout {
    anchors.fill: parent; columns: 2; columnSpacing: 16; rowSpacing: 12

    NeonCard {
      Layout.fillWidth: true; Layout.preferredHeight: 160
      Column {
        anchors.fill: parent; spacing: 8
        Label { text: "Audio Output"; color: "#D7FFF9"; font.bold: true }
        ComboBox { model: ["Default"] /* TODO: lista dispositivi QT */ }
        Label { text: "Sample rate: 24 kHz mono s16le"; color: "#9BD7FF" }
      }
    }

    NeonCard {
      Layout.fillWidth: true; Layout.preferredHeight: 160
      Column {
        anchors.fill: parent; spacing: 8
        Label { text: "Server Realtime"; color: "#D7FFF9"; font.bold: true }
        Label { text: "Protocollo: hello/partial/answer + binario PCM"; color: "#9BD7FF" }
        Label { text: "Barge-in supportato"; color: "#9BD7FF" }
      }
    }

    NeonCard {
      Layout.columnSpan: 2; Layout.fillWidth: true; Layout.preferredHeight: 120
      Column {
        anchors.fill: parent; spacing: 6
        Label { text: "Aspetto"; color: "#D7FFF9"; font.bold: true }
        Row { spacing: 8
          Button { text: "Tema scuro (default)"; }
          Button { text: "Tema blu-neon"; }
        }
      }
    }
  }
}
