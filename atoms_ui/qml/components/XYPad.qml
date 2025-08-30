import QtQuick 2.15
import ".." as Theme

Item {
    id: root
    property real xValue: 0.5
    property real yValue: 0.5
    signal positionChanged(real x, real y)

    width: 150; height: 150

    Rectangle {
        anchors.fill: parent
        color: Theme.Theme.panel
        border.color: Theme.Theme.border
    }

    Rectangle {
        id: handle
        width: 14; height: 14
        radius: 7
        color: Theme.Theme.teal
        x: root.xValue * (parent.width - width)
        y: root.yValue * (parent.height - height)
    }

    MouseArea {
        anchors.fill: parent
        onPositionChanged: update(mouse.x, mouse.y)
        onPressed: update(mouse.x, mouse.y)
        function update(mx, my) {
            root.xValue = Math.max(0, Math.min(1, mx/parent.width));
            root.yValue = Math.max(0, Math.min(1, my/parent.height));
            handle.x = root.xValue * (parent.width - handle.width);
            handle.y = root.yValue * (parent.height - handle.height);
            root.positionChanged(root.xValue, root.yValue);
        }
    }
}
