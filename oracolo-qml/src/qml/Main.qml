import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import "./Palette.js" as Palette
import "components" as Components
import "components"


ApplicationWindow {
  id: win
  width: 1200
  height: 800
  visible: true
  title: "Occhio Onniveggente · Oracolo ✨"

  // Palette base
  property color bg: Palette.bg
  property color card: Palette.card
  property color accent: Palette.accent
  property color text: Palette.text

  // Background gradient with subtle star field
  Rectangle {
    anchors.fill: parent
    gradient: Gradient {
      GradientStop { position: 0.0; color: Qt.darker(Palette.accent, 3) }
      GradientStop { position: 1.0; color: Palette.bg }
    }
  }

  Canvas {
    anchors.fill: parent
    opacity: 0.15
    onPaint: {
      var ctx = getContext("2d");
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = Palette.text;
      for (var i = 0; i < 100; ++i) {
        var x = Math.random() * width;
        var y = Math.random() * height;
        var r = Math.random() * 1.5;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    Component.onCompleted: requestPaint()
  }

  Components.OrbGlow {
    radius: 180
    anchors.centerIn: parent
  }


  // Sidebar semplificata con OrbGlow e waveform
  Column {

  /* Sidebar (Modalità)
  Rectangle {
  
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

    Label {
      text: "Wake Word Recognized"
      color: text
      width: parent.width
      horizontalAlignment: Text.AlignHCenter
      font.pixelSize: 16

    }
  }
  */

  // Tabs top
  TabBar {
    id: tabs
    anchors { left: parent.left; right: parent.right; top: parent.top }
    TabButton { text: "Chat" }
    TabButton { text: "Documenti" }
    TabButton { text: "Impostazioni" }
  }

  // Contenuto (tabs)
  StackLayout {
    id: stack
    anchors { left: parent.left; right: parent.right; top: tabs.bottom; bottom: parent.bottom; margins: 16 }
    currentIndex: tabs.currentIndex

    ChatPage { }
    DocumentsPage { }
    SettingsPage { }
  }
}
