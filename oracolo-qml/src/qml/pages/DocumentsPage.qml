import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../Palette.js" as Palette

Item {
  anchors.fill: parent

  ColumnLayout {
    anchors.fill: parent; spacing: 12

    NeonCard {
      Layout.fillWidth: true; Layout.preferredHeight: 100
      color: Palette.card
      Column {
        anchors.fill: parent; spacing: 8
        Label { text: "Gestione Documenti & Regole"; color: Palette.text; font.bold: true }
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
      color: Palette.card
      Column {
        anchors.fill: parent; spacing: 8
        Label { text: "Regole/Topic correnti (UI only)"; color: Palette.accentLight }
        TextArea {
          anchors.left: parent.left; anchors.right: parent.right; anchors.bottom: parent.bottom; anchors.top: previous.bottom
          placeholderText: "Scrivi keywords o policy di dominio…"
        }
      }
    }
  }
}
