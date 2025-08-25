import QtQuick 2.15
import Theme 1.0

Text {
    id: root
    text: "TAB"
    signal clicked
    color: mouse.hovered ? Theme.neonA : Theme.text
    font.family: Theme.font
    font.pixelSize: 16
    MouseArea {
        id: mouse
        anchors.fill: parent
        hoverEnabled: true
        onClicked: root.clicked()
    }
}
