import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components"

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

  // Sidebar semplificata con OrbGlow e waveform
  Column {
    id: sidebar
    width: 240
    anchors.top: parent.top
    anchors.bottom: parent.bottom
    anchors.left: parent.left
    spacing: 16
    padding: 16

    OrbGlow {
      id: orb
      width: 200
      height: 200
      anchors.horizontalCenter: parent.horizontalCenter
    }

    Waveform {
      id: wf
      width: parent.width - 4
      height: 60
      colorLine: accent
      Component.onCompleted: start()
      onTick: appendLevel(rt.level)
    }

    Label {
      text: "Wake Word Recognized"
      color: text
      width: parent.width
      horizontalAlignment: Text.AlignHCenter
      font.pixelSize: 16
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
