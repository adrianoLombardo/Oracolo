import QtQuick 2.15
import ".."
import Theme 1.0
import "."

ListView {
    id: root
    property var historyModel
    width: parent ? parent.width : 300
    model: historyModel
    spacing: 10
    clip: true

    delegate: ChatBubble {
        width: root.width
        text: model.content
        fromUser: model.role === "user"
    }
}
