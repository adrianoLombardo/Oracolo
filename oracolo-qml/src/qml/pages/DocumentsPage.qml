import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
  anchors.fill: parent

  ColumnLayout {
    anchors.fill: parent; spacing: 12

    NeonCard {
      Layout.fillWidth: true; Layout.preferredHeight: 100
      Column {
        anchors.fill: parent; spacing: 8
        Label { text: "Gestione Documenti & Regole"; color: "#D7FFF9"; font.bold: true }
        Row {
          spacing: 8
          Button { text: "Aggiungi documenti…"; onClicked: console.log("TODO: apri file dialog e invia ingest al backend") }
          Button { text: "Rimuovi…" }
          Button { text: "Aggiorna indice" }
        }
      }
    }

    NeonCard {
      Layout.fillWidth: true; Layout.fillHeight: true
      Column {
        anchors.fill: parent; spacing: 8
        Label { text: "Regole/Topic correnti (UI only)"; color: "#66FFF2" }
        TextArea {
          anchors.left: parent.left; anchors.right: parent.right; anchors.bottom: parent.bottom; anchors.top: previous.bottom
          placeholderText: "Scrivi keywords o policy di dominio…"
        }
      }
    }
  }
}
