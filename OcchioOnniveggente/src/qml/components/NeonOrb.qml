import QtQuick 2.15
import ".."
import Theme 1.0


Rectangle {
    id: orb
    width: 200
    height: 200
    radius: width/2
    gradient: Gradient {
        GradientStop { position: 0.0; color: Theme.neonA }
        GradientStop { position: 1.0; color: Theme.neonB }
    }
    border.color: Theme.neonA
    border.width: Theme.borderW
}
