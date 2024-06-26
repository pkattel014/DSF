using ITensors, DataFrames, Printf, ITensors.HDF5

N=100
Jz= 1
D = 0
println("N= ",N);
println("Jz= ", Jz);
println("D= ",D);

sites = siteinds("S=1", N, conserve_qns=true)
function create_os(N)
os = OpSum()
    for j = 1:N-1
             os += 1.0, "Sz", j, "Sz", j + 1
             os += 0.5, "S+", j, "S-", j + 1
             os += 0.5, "S-", j, "S+", j + 1
             os += D, "Sz2" , j
    end
            os += D, "Sz2", N

    return os
end


os = create_os(N)

sites = siteinds("S=1", N, conserve_qns=true)
H = MPO(os, sites)

state = [isodd(n) ? "Up" : "Dn" for n=1:N]
psi0_init = productMPS(sites, state)

nsweeps = 20
maxdim = [10,20,50,100,100,200,200,200,500,500,500,1000,1000,1000,1000,1000,2000]
cutoff = [1E-10]
noise = [1E-6,1E-6,1E-7,1E-8,0,0,0,0]

energy, psi = dmrg(H, psi0_init; nsweeps, maxdim, cutoff, noise)
println("\nGround state energy = ",energy)
println("\nTotal Sz of Ground State = ",sum(expect(psi,"Sz")))
println("\noverlap with the initial state = ",inner(psi0_init,psi))


tstep = 0.02
cutoff = 1E-8;

gates = ITensor[]
    for j in 1:(N - 1)
        s1 = sites[j]
        s2 = sites[j + 1]
        hj =Jz * op("Sz", s1) * op("Sz", s2) + 1/2 * op("S+", s1) * op("S-", s2) + 1/2 * op("S-", s1) * op("S+", s2)
       Gj = exp(-im * tstep / 2 * hj)
        push!(gates, Gj)
    end
    for j in 1:N
            s1 = sites[j]
        qj=D*op("Sz2",s1)
        Qj = exp(-im * tstep / 2 * qj)
        push!(gates, Qj)
    end
       	append!(gates, reverse(gates))


psi0 = deepcopy(psi)



sop = AutoMPO()
sop += 1,"Sz",div(N,2);
Operator= MPO(sop,sites);
finalpsi=apply(Operator,psi)
sp_psi=noprime(finalpsi)

println("Computing dynamics")

data=[[],[],[],[]]

ft= 1200
for t in 1:ft
    global sp_psi
    sp_psi = apply(gates, sp_psi; cutoff)
    for n in 1:N
        op = AutoMPO()
        op += 1,"Sz",n;
        operator= MPO(op,sites);
    res=inner(psi0',operator,sp_psi)
    res = res.*exp(im*energy*t*tstep)
    push!(data[1], t*tstep)
    push!(data[2], n)
    push!(data[3], real(res))
    push!(data[4], imag(res))
    end
       	
end

udata=hcat(data...)

cdata=udata |> x -> convert(Matrix{Float64}, x)
h5write("data_Jz=$(Jz)_D=$(D).h5", "dataset", cdata)
