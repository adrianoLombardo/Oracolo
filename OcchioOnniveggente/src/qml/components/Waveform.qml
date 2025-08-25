import QtQuick 2.15
import ".."
import Theme 1.0

Canvas {
    id: root
    onPaint: {
        var ctx = getContext("2d");
        ctx.reset();
        ctx.lineWidth = 2;
        ctx.strokeStyle = Theme.neonA;
        ctx.beginPath();
        var h = height / 2;
        for (var x = 0; x < width; x++) {
            var y = h + Math.sin(x / width * Math.PI * 2) * (h - 4);
            if (x === 0)
                ctx.moveTo(x, y);
            else
                ctx.lineTo(x, y);
        }
        ctx.stroke();
    }
}
