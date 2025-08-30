import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import "." as Theme
import "components"

ApplicationWindow {
    id: win
    width: 1280
    height: 900
    visible: true
    color: Theme.Theme.bg
    title: "Atoms"

    Rectangle {
        id: topBar
        anchors.left: parent.left
        anchors.right: parent.right
        height: 40
        color: Theme.Theme.panel
        RowLayout {
            anchors.fill: parent
            spacing: 10
            Text { text: "Physical Modeling Synth"; color: Theme.Theme.text; font.pixelSize: 14; Layout.leftMargin: 20 }
            Item { Layout.fillWidth: true }
            Text { text: "MPE"; color: Theme.Theme.teal; font.pixelSize: 12 }
            Image { source: "../assets/icons/play.svg"; width: 20; height: 20; fillMode: Image.PreserveAspectFit }
            Layout.alignment: Qt.AlignVCenter
        }
    }

    AtomView {
        id: atom
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
        width: 400; height: 400
    }

    // Knobs around
    Knob { label: "Chaos"; anchors.top: atom.top; anchors.horizontalCenter: atom.left; anchors.topMargin: -20; anchors.horizontalCenterOffset: -120 }
    Knob { label: "Force"; anchors.left: atom.left; anchors.verticalCenter: atom.verticalCenter; anchors.leftMargin: -150 }
    Knob { label: "Drive"; anchors.bottom: atom.bottom; anchors.horizontalCenter: atom.left; anchors.bottomMargin: -20; anchors.horizontalCenterOffset: -120 }
    Knob { label: "Order"; anchors.top: atom.top; anchors.horizontalCenter: atom.right; anchors.topMargin: -20; anchors.horizontalCenterOffset: 120 }
    Knob { label: "Overtones"; anchors.left: atom.right; anchors.verticalCenter: atom.verticalCenter; anchors.leftMargin: 150 }
    Knob { label: "Filter"; anchors.bottom: atom.bottom; anchors.horizontalCenter: atom.right; anchors.bottomMargin: -20; anchors.horizontalCenterOffset: 120 }

    // Bottom bar
    Rectangle {
        id: bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 140
        color: Theme.Theme.panel
        RowLayout {
            anchors.fill: parent
            anchors.margins: 20
            spacing: 40
            ArcSlider { Layout.preferredWidth: 100; Layout.alignment: Qt.AlignBottom; label: "Attack" }
            ArcSlider { Layout.preferredWidth: 100; Layout.alignment: Qt.AlignBottom; label: "Release"; color: Theme.Theme.purple }
            ArcSlider { Layout.preferredWidth: 100; Layout.alignment: Qt.AlignBottom; label: "Movement"; color: Theme.Theme.lime }
            ArcSlider { Layout.preferredWidth: 100; Layout.alignment: Qt.AlignBottom; label: "Modulation"; color: Theme.Theme.cyan }
            ArcSlider { Layout.preferredWidth: 100; Layout.alignment: Qt.AlignBottom; label: "Vibrato"; color: Theme.Theme.orange }
            XYPad { Layout.fillWidth: true; Layout.preferredHeight: 120 }
        }
    }
}
