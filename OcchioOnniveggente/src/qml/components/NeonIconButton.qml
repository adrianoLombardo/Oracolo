import QtQuick 2.15
import ".."
import Theme 1.0


Rectangle {
    id: root
    property string iconText: ""
    signal clicked
    width: 56
    height: 56
    radius: width/2
    color: Theme.panel
    border.color: Theme.neonA
    border.width: Theme.borderW

    Text {
        id: icon
        text: root.iconText
        anchors.centerIn: parent
        color: Theme.neonA
        font.pixelSize: 24
        font.family: Theme.font
    }

    MouseArea {
        anchors.fill: parent
        onClicked: root.clicked()
        hoverEnabled: true
    }
}
