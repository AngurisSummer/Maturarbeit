let Br = 1.345
let const_mag = -Br/(4*Math.PI)

class Magnet {
    constructor(a,b,c, x, y, z){
        this.a = a
        this.b = b
        this.c = c
        this.x = x
        this.y = y
        this.z = z
    }
    calculateF(){
        let xna = this.x+this.a
        let ynb = this.y+this.b
        let znc = this.z+this.c
        let dist = Math.sqrt(xna*xna + ynb*ynb + znc*znc)
        let temp = (xna*ynb)/(znc*dist)
        return Math.atan(temp)
    }
    calculateB(){
        let c1 = this.calculateF(-this.x,this.y,this.z)
        let c2 = this.calculateF(this.x,-this.y,this.z)
        let c3 = this.calculateF(this.x,this.y,-this.z)
        let c4 = this.calculateF(-this.x,-this.y,this.z)
        let c5 = this.calculateF(-this.x,this.y,-this.z)
        let c6 = this.calculateF(this.x,-this.y,-this.z)
        let c7 = this.calculateF(-this.x,-this.y,-this.z)
        let c8 = this.calculateF(this.x,this.y,this.z)
        let sum = c1+c2+c3+c4+c5+c6+c7+c8
        return const_mag*sum
    }
}

