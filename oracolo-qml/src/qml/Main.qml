import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects

ApplicationWindow {
  id: win
  width: 1200
  height: 800
  visible: true
  title: "Occhio Onniveggente · Oracolo ✨"

  // Palette base
  property color bg: "#0B1020"
  property color card: "#10182A"
  property color accent: "#00E5FF"
  property color text: "#D7FFF9"

  Rectangle { anchors.fill: parent; color: bg }

  // Sidebar (Modalità)
  Rectangle {
    id: sidebar
    width: 240; anchors.top: parent.top; anchors.bottom: parent.bottom; anchors.left: parent.left
    color: "#0E1422"
    layer.enabled: true
    layer.effect: MultiEffect {
      shadowEnabled: true; shadowColor: "#00131A"; shadowBlur: 0.6
      brightness: 0.02
    }

    Column {
      anchors.fill: parent; anchors.margins: 16; spacing: 12
      Label { text: "Modalità"; color: text; font.pixelSize: 16; font.bold: true }
      ButtonGroup { id: modeGroup }
      Repeater {
        model: ["Museo", "Galleria", "Conferenze", "Didattica"]
        delegate: RadioButton {
          text: modelData; checked: index === 0; width: parent.width
          palette.buttonText: text
          onToggled: if (checked) {
            // TODO: invia al backend (via REST o WS di controllo) il topic/mode
            console.log("Mode set:", text)
          }
        }
      }

      // connessione WS
      Rectangle { height: 1; width: parent.width; color: "#1B263B"; opacity: 0.6 }
      TextField {
        id: urlEdit; placeholderText: "ws://127.0.0.1:8765"
        text: "ws://127.0.0.1:8765"; color: text; background: Rectangle{ color:"#0C1424"; radius: 8 }
      }
      Row {
        spacing: 8
        Button {
          text: rt.connected ? "Disconnetti" : "Connetti"
          onClicked: rt.connected ? rt.disconnectFromServer() : rt.connectToUrl(urlEdit.text)
        }
        Rectangle {
          width: 12; height: 12; radius: 6
          color: rt.connected ? "#29FF92" : "#FF4D4F"
          border.color: "#002A1A"; border.width: 1
        }
      }

      // VU meter esempio
      Rectangle { height: 1; width: parent.width; color: "#1B263B"; opacity: 0.6 }
      Label { text: "Livello Audio"; color: text }
      Rectangle {
        width: parent.width - 4; height: 16; radius: 8; color: "#0C1424"
        Rectangle {
          anchors.verticalCenter: parent.verticalCenter
          height: parent.height
          width: parent.width * rt.level
          radius: 8
          gradient: Gradient {
            GradientStop { position: 0.0; color: "#00E5FF" }
            GradientStop { position: 1.0; color: "#66FFF2" }
          }
        }
      }

      // Waveform semplice (usa rt.level storico)
      Waveform {
        id: wf
        width: parent.width - 4; height: 60
        colorLine: accent
        Component.onCompleted: start()
        onTick: appendLevel(rt.level)
      }

      DocumentsPanel {
        width: parent.width - 4
        onDocumentSelected: function(name) {
          if (rt.connected) {
            rt.sendDocument(name)
          } else {
            Qt.openUrlExternally(Qt.resolvedUrl(name))
          }
        }
      }
    }
  }

  // Contenuto (tabs)
  StackLayout {
    id: stack
    anchors { left: sidebar.right; right: parent.right; top: parent.top; bottom: parent.bottom; margins: 16 }
    currentIndex: tabs.currentIndex

    ChatPage { }
    DocumentsPage { }
    SettingsPage { }
  }

  // Tabs top
  TabBar {
    id: tabs
    anchors { left: sidebar.right; right: parent.right; top: parent.top; margins: 16 }
    TabButton { text: "Chat" }
    TabButton { text: "Documenti" }
    TabButton { text: "Impostazioni" }
  }
}
