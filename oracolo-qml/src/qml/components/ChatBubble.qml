import QtQuick
import QtQuick.Controls

Item {
    id: root
    property alias text: message.text
    property bool fromUser: false

    width: ListView.view ? ListView.view.width : 360
    implicitHeight: bubble.height + 8

    Rectangle {
        id: bubble
        radius: 8
        color: root.fromUser ? "#66FFF2" : "#333333"
        anchors.left: root.fromUser ? undefined : parent.left
        anchors.right: root.fromUser ? parent.right : undefined
        anchors.margins: 8
        width: Math.min(message.paintedWidth + 24, root.width * 0.8)
        height: message.paintedHeight + 16

        Text {
            id: message
            anchors.fill: parent
            anchors.margins: 8
            wrapMode: Text.Wrap
            color: root.fromUser ? "#000000" : "#D7FFF9"
        }
    }
}
