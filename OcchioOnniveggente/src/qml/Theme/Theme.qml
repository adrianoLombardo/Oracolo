pragma Singleton
import QtQuick 2.15

QtObject {
    // Palette ispirata allo screenshot
    readonly property color bg:          "#0b0f1a"
    readonly property color panel:       "#0f1422"
    readonly property color border:      "#1a2640"
    readonly property color neonA:       "#31d3ff"   // ciano
    readonly property color neonB:       "#a855f7"   // viola
    readonly property color text:        "#9fe8ff"
    readonly property color textSoft:    "#79b7c9"
    readonly property color bubbleUser:  "#0f1b2e"
    readonly property color bubbleBot:   "#0f2530"
    readonly property real  radius:      18
    readonly property real  borderW:     1.5
    readonly property string font:       "Inter"
}
