import QtQuick 2.15
import ".."

Rectangle {
    id: root
    property bool fromUser: false
    property string text: ""
    color: fromUser ? Theme.bubbleUser : Theme.bubbleBot
    radius: Theme.radius
    border.color: Theme.border
    border.width: Theme.borderW
    width: parent ? parent.width * 0.9 : 300
    x: fromUser && parent ? parent.width - width : 0

    Text {
        id: label
        text: root.text
        wrapMode: Text.Wrap
        anchors.fill: parent
        anchors.margins: 12
        color: Theme.text
        font.family: Theme.font
        font.pixelSize: 14
    }
}
