/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2014 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "kOmegaSSTx.H"
#include "addToRunTimeSelectionTable.H"

#include "backwardsCompatibilityWallFunctions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace incompressible
{
namespace RASModels
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(kOmegaSSTx, 0);
addToRunTimeSelectionTable(RASModel, kOmegaSSTx, dictionary);

// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

tmp<volScalarField> kOmegaSSTx::F1(const volScalarField& CDkOmega) const
{
    tmp<volScalarField> CDkOmegaPlus = max
    (
        CDkOmega,
        dimensionedScalar("1.0e-10", dimless/sqr(dimTime), 1.0e-10)
    );

    tmp<volScalarField> arg1 = min
    (
        min
        (
            max
            (
                (scalar(1)/betaStar_)*sqrt(k_)/(omega_*y_),
                scalar(500)*nu()/(sqr(y_)*omega_)
            ),
            (4*alphaOmega2_)*k_/(CDkOmegaPlus*sqr(y_))
        ),
        scalar(10)
    );

    return tanh(pow4(arg1));
}


tmp<volScalarField> kOmegaSSTx::F2() const
{
    tmp<volScalarField> arg2 = min
    (
        max
        (
            (scalar(2)/betaStar_)*sqrt(k_)/(omega_*y_),
            scalar(500)*nu()/(sqr(y_)*omega_)
        ),
        scalar(100)
    );

    return tanh(sqr(arg2));
}


tmp<volScalarField> kOmegaSSTx::F3() const
{
    tmp<volScalarField> arg3 = min
    (
        150*nu()/(omega_*sqr(y_)),
        scalar(10)
    );

    return 1 - tanh(pow4(arg3));
}


tmp<volScalarField> kOmegaSSTx::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23() *= F3();
    }

    return f23;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

kOmegaSSTx::kOmegaSSTx
(
    const volVectorField& U,
    const surfaceScalarField& phi,
    transportModel& transport,
    const word& turbulenceModelName,
    const word& modelName
)
:
    RASModel(modelName, U, phi, transport, turbulenceModelName),

    alphaK1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaK1",
            coeffDict_,
            0.85
        )
    ),
    alphaK2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaK2",
            coeffDict_,
            1.0
        )
    ),
    alphaOmega1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaOmega1",
            coeffDict_,
            0.5
        )
    ),
    alphaOmega2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaOmega2",
            coeffDict_,
            0.856
        )
    ),
    gamma1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "gamma1",
            coeffDict_,
            5.0/9.0
        )
    ),
    gamma2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "gamma2",
            coeffDict_,
            0.44
        )
    ),
    beta1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "beta1",
            coeffDict_,
            0.075
        )
    ),
    beta2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "beta2",
            coeffDict_,
            0.0828
        )
    ),
    betaStar_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "betaStar",
            coeffDict_,
            0.09
        )
    ),
    a1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "a1",
            coeffDict_,
            0.31
        )
    ),
    b1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "b1",
            coeffDict_,
            1.0
        )
    ),
    c1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "c1",
            coeffDict_,
            10.0
        )
    ),
    F3_
    (
        Switch::lookupOrAddToDict
        (
            "F3",
            coeffDict_,
            false
        )
    ),

    y_(mesh_),

    k_
    (
        IOobject
        (
            "k",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        autoCreateK("k", mesh_)
    ),
    omega_
    (
        IOobject
        (
            "omega",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        autoCreateOmega("omega", mesh_)
    ),
    nut_
    (
        IOobject
        (
            "nut",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        autoCreateNut("nut", mesh_)
    ),
    p_
    (
        IOobject
        (
            "p",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        autoCreateNut("p", mesh_)
        //mesh_ 
    ),
     R_
    (
        IOobject
        (
            "R",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        0.0*k_*symm(fvc::grad(U_)/omega_)
    ),
    Ax_
    (
        IOobject
        (
            "Ax",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        0.0*symm(fvc::grad(U_))/omega_
    ),
    Rx_
    (
        IOobject
        (
            "Rx",
            runTime_.timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        0.0*symm(fvc::grad(U_))/omega_
    )
{
    bound(k_, kMin_);
    bound(omega_, omegaMin_);

    nut_ =
    (
        a1_*k_
      / max
        (
            a1_*omega_,
            b1_*F23()*sqrt(2.0)*mag(symm(fvc::grad(U_)))
        )
    );
    nut_.correctBoundaryConditions();

    printCoeffs();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

tmp<volSymmTensorField> kOmegaSSTx::R() const
{
    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                "R",
                runTime_.timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            ((2.0/3.0)*I)*k_ - nut_*twoSymm(fvc::grad(U_)) + 2*k_*Ax_,
            k_.boundaryField().types()
        )
    );
}


tmp<volSymmTensorField> kOmegaSSTx::devReff() const
{
    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                "devRhoReff",
                runTime_.timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
           -nuEff()*dev(twoSymm(fvc::grad(U_))) + dev(2*k_*Ax_)
        )
    );
}


tmp<fvVectorMatrix> kOmegaSSTx::divDevReff(volVectorField& U) const
{
    return
    (
      - fvm::laplacian(nuEff(), U)
      - fvc::div(nuEff()*dev(T(fvc::grad(U))))
      + fvc::div(dev(2*k_*Ax_))
    );
}


tmp<fvVectorMatrix> kOmegaSSTx::divDevRhoReff
(
    const volScalarField& rho,
    volVectorField& U
) const
{
    volScalarField muEff("muEff", rho*nuEff());

    return
    (
      - fvm::laplacian(muEff, U)
      - fvc::div(muEff*dev(T(fvc::grad(U))))
      + fvc::div(dev(2*rho*k_*Ax_))
    );
}


bool kOmegaSSTx::read()
{
    if (RASModel::read())
    {
        alphaK1_.readIfPresent(coeffDict());
        alphaK2_.readIfPresent(coeffDict());
        alphaOmega1_.readIfPresent(coeffDict());
        alphaOmega2_.readIfPresent(coeffDict());
        gamma1_.readIfPresent(coeffDict());
        gamma2_.readIfPresent(coeffDict());
        beta1_.readIfPresent(coeffDict());
        beta2_.readIfPresent(coeffDict());
        betaStar_.readIfPresent(coeffDict());
        a1_.readIfPresent(coeffDict());
        b1_.readIfPresent(coeffDict());
        c1_.readIfPresent(coeffDict());
        F3_.readIfPresent("F3", coeffDict());

        return true;
    }
    else
    {
        return false;
    }
}


void kOmegaSSTx::correct()
{
    RASModel::correct();

    if (!turbulence_)
    {
        return;
    }

    if (mesh_.changing())
    {
        y_.correct();
    }

    // Calculate basis tensors and invariants
    volTensorField gradU = fvc::grad(U_);
    volSymmTensorField Sij = dev(symm(gradU));
    volTensorField Wij = -0.5*(gradU - gradU.T());

    volScalarField S = sqrt(2*magSqr(symm(gradU)));
    volScalarField tau = 1./max(S/a1_ + omegaMin_, omega_ + omegaMin_);                         // Check if limiter is necessary

    volSymmTensorField sij = Sij * tau;
    volTensorField wij = Wij * tau;

    volScalarField I01 = tr(sij & sij);
    volScalarField I02 = tr(wij & wij);
	volScalarField I03 = tr(sij & (sij & sij));
	volScalarField I04 = tr(wij & (wij & sij));
	volScalarField I05 = tr(wij & (wij & (sij & sij)));

    volSymmTensorField T01 = sij;
    volSymmTensorField T02 = symm((sij & wij) - (wij & sij));
    volSymmTensorField T03 = symm(sij & sij) - scalar(1.0/3.0)*I*I01;
    volSymmTensorField T04 = symm(wij & wij) - scalar(1.0/3.0)*I*I02;
	volSymmTensorField T05 = symm((wij & (sij & sij)) - ((sij & sij) & wij));
	volSymmTensorField T06 = symm(((wij & wij) & sij) + ((sij & sij) & wij) - (2.0/3.0) * I * tr(sij & (wij & wij)));
	volSymmTensorField T07 = symm((wij & (sij & (wij & wij))) - (wij & (wij & (sij & wij))));
	volSymmTensorField T08 = symm((sij & (wij & (sij & sij))) - (sij & (sij & (wij & sij))));
	volSymmTensorField T09 = symm((wij & (wij & (sij & sij))) + (sij & (sij & (wij & wij))) - (2.0/3.0) * I * tr(sij & (sij & (wij & wij))));
	volSymmTensorField T10 = symm((wij & (sij & (sij & (wij & wij)))) -(wij & (wij & (sij & (sij & wij)))));

    dimensionedScalar kSMALL("0",dimLength*dimLength/dimTime/dimTime, 1e-10);
    dimensionedScalar rSMALL ("0", dimensionSet(0,2,-2,0,0,0,0),1e-10);
    dimensionedScalar pSMALL ("0", dimensionSet(0,2,-2,0,0,0,0),1e-10);
    dimensionedScalar USMALL ("0", dimensionSet(0,1,-1,0,0,0,0),1e-10);
    dimensionedScalar qSMALL ("0", dimensionSet(0,0,-1,0,0,0,0),1e-10);
    dimensionedScalar NewSMALL("0", dimensionSet(0,0,0,0,0,0,0), 1e-10);
    dimensionedScalar constantInRe("0", dimensionSet(0,0,0,0,0,0,0), 2);
    dimensionedScalar osmall ("0",dimensionSet(0,0,-1,0,0,0,0),1e-10);
    wallDist y(mesh_);

    //tau = 1.0 / (omega + osmall);
    volScalarField ell = sqrt(mag(k_)) * tau ;

    // Q1
    Info<< "    Calculating Q1    " << endl;
    volScalarField Q1_org =  0.5*(Foam::sqr(tr(gradU)) - tr(((gradU) & (gradU))));
    volScalarField Q01 = (Q1_org-Foam::min(Q1_org))/(Foam::max(Q1_org)-Foam::min(Q1_org));
//  volScalarField Q01 = ell / (y+ell); // NEW
//    volScalarField Q01 (
//        IOobject (
//            "Q01",
//            runTime_.timeName(),
//            mesh_,
//            IOobject::NO_READ
//        ),
//        (Q1_org-Foam::min(Q1_org))/(Foam::max(Q1_org)-Foam::min(Q1_org)),
//        "zeroGradient"
//    );
      Info << "--> Q1_org Min:" << Foam::min(Q1_org).value() << " Max: " << Foam::max(Q1_org).value() << endl;
      Info << "--> Q1     Min:" << Foam::min(Q01).value()    << " Max: " << Foam::max(Q01).value()    << endl;


    // Q2
    Info<< "    Calculating Q2    " << endl;
    volScalarField Q02 = k_/ (0.5* (U_&U_) + k_ + kSMALL);
//    volScalarField Q02 (
//            IOobject (
//                    "Q02",
//                    runTime_.timeName(),
//                    mesh_,
//                    IOobject::NO_READ
//            ),
//            k_/ (0.5* (U_&U_) + k_ + kSMALL),
//            "zeroGradient"
//    );
      Info << "--> Q2    Min:" << Foam::min(Q02).value() << " Max :" << Foam::max(Q02).value() << endl;
              

    Info<< "    Reading transport Properties" <<endl;
    IOdictionary transportProperties
    (
        IOobject
        (
            "transportProperties",
            runTime_.constant(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false
        )
    );

    dimensionedScalar nu (transportProperties.lookup("nu"));
    Info << "--> nu  :" << nu.value() << endl;

    // Q3
    Info<< "    Calculating Q3    " << endl;
    volScalarField Q03 = mag(y * Foam::sqrt(k_+kSMALL) / nu)/10000;
//    volScalarField Q03 (
//        IOobject (
//            "Q03",
//            runTime_.timeName(),
//            mesh_,
//            IOobject::NO_READ
//        ),
//        mag(y * Foam::sqrt(k_+kSMALL) / nu)/10000,
//        "zeroGradient"
//    );
      Info << "--> Q3    Min:" << Foam::min(Q03).value() << " Max :" << Foam::max(Q03).value() << endl;

     // Q4
     Info<< "    Calculating Q4    " << endl;
     const volScalarField& p_ = mesh_.objectRegistry::template lookupObject<volScalarField>("p");
     volVectorField gradp = fvc::grad(p_);
     //volVectorField gradp = fvc::grad(p_+pSMALL);
     Info << "--> gradp:" << Foam::min(gradp).value() << " Max: " << Foam::max(gradp).value() << endl;
     volScalarField Q4_org = U_ & gradp;
     volScalarField Q04 = (Q4_org-Foam::min(Q4_org))/(Foam::max(Q4_org)-Foam::min(Q4_org));
////    volScalarField Q04 (
////        IOobject (
////            "Q04",
////            runTime_.timeName(),
////            mesh_,
////            IOobject::NO_READ
////        ),
////        //(Q4_org-Foam::min(Q4_org))/(Foam::max(Q4_org)-Foam::min(Q4_org)),
////        //"zeroGradient"
////        //dimless,
////        mesh_
////    );
     Info << "--> Q4_org Min:" << Foam::min(Q4_org).value() << " Max: " << Foam::max(Q4_org).value() << endl;
     Info << "--> Min Q4:"     << Foam::min(Q04).value()    << "Max :"  << Foam::max(Q04).value()    << endl;


    // Q5
    Info<< "    Calculating Q5    " << endl;
    volScalarField Q5_org = mag(fvc::curl(U_));
    volScalarField Q05 = (Q5_org-Foam::min(Q5_org))/(Foam::max(Q5_org)-Foam::min(Q5_org));
//    volScalarField Q05 (
//            IOobject (
//                "Q05",
//                runTime_.timeName(),
//                mesh_,
//                IOobject::NO_READ
//            ),
//            (Q5_org-Foam::min(Q5_org))/(Foam::max(Q5_org)-Foam::min(Q5_org)),
//            "zeroGradient"
//        );
    Info << "--> Q5_org Min:" << Foam::min(Q5_org).value() << " Max: " << Foam::max(Q5_org).value() << endl;
    Info << "--> Min Q5:"     << Foam::min(Q05).value()    <<  "Max :" << Foam::max(Q05).value()    << endl;

    // Q6
    Info<< "    Calculating Q6    " << endl;
    volScalarField Q06 = nut_ / (1. * nu*100 + nut_);
//    volScalarField Q06 (
//    IOobject (
//                "Q06",
//                runTime_.timeName(),
//                mesh_,
//                IOobject::NO_READ
//            ),
//            nut_ / (1. * nu*100 + nut_),
//            "zeroGradient"
//    );
    Info << "--> Q6    Min:" << Foam::min(Q06).value() << " Max :" << Foam::max(Q06).value() << endl;

    // Q7
    Info<< "    Calculating Q7    " << endl;
    volScalarField Q7_org = Foam::sqrt(gradp & gradp);
    volScalarField Q07 = (Q7_org-Foam::min(Q7_org))/(Foam::max(Q7_org)-Foam::min(Q7_org));
////    volScalarField Q07 (
////    IOobject (
////                "Q07",
////                runTime_.timeName(),
////                mesh_,
////                IOobject::NO_READ
////            ),
////            //(Q7_org-Foam::min(Q7_org))/(Foam::max(Q7_org)-Foam::min(Q7_org)),
////            //"zeroGradient"
////            //dimless,
////            mesh_
////    );
     Info << "--> Q7_org Min:" << Foam::min(Q7_org).value() << " Max: " << Foam::max(Q7_org).value() << endl;
     Info << "--> Min Q7:"     << Foam::min(Q07).value()    <<  "Max :" << Foam::max(Q07).value()    << endl;

    // Q8
    Info<< "    Calculating Q8    " << endl;
    volScalarField Q8_org = mag((U_ * U_) && gradU);
    volScalarField Q08 = (Q8_org-Foam::min(Q8_org))/(Foam::max(Q8_org)-Foam::min(Q8_org));
//    volScalarField Q08 (
//            IOobject (
//                "Q08",
//                runTime_.timeName(),
//                mesh_,
//                IOobject::NO_READ
//            ),
//            (Q8_org-Foam::min(Q8_org))/(Foam::max(Q8_org)-Foam::min(Q8_org)),
//            "zeroGradient"
//     );
    Info << "--> Q8_org Min:" << Foam::min(Q8_org).value() << " Max: " << Foam::max(Q8_org).value() << endl;
    Info << "--> Min/Max Q8:" << Foam::min(Q08).value()    <<  Foam::max(Q08).value()               << endl;

    // Q9
    Info<< "    Calculating Q9    " << endl;
    volScalarField Q9_org = (fvc::grad(k_) && U_);
    volScalarField Q09 = (Q9_org-Foam::min(Q9_org))/(Foam::max(Q9_org)-Foam::min(Q9_org));
//    volScalarField Q09 (
//            IOobject (
//                "Q09",
//                runTime_.timeName(),
//                mesh_,
//                IOobject::NO_READ
//            ),
//            (Q9_org-Foam::min(Q9_org))/(Foam::max(Q9_org)-Foam::min(Q9_org)),
//           "zeroGradient"
//    );
    Info << "--> Q9_org Min:" << Foam::min(Q9_org).value() << " Max: " << Foam::max(Q9_org).value() << endl;
    Info << "--> Min/Max Q9:" << Foam::min(Q09).value()    << Foam::max(Q09).value()                << endl;

//    Info<< "    Reading the Reynolds stress : R" <<endl;
//    IOobject Rheader (
//            "R",
//            runTime_.timeName(),
//            mesh_,
//            IOobject::NO_READ
//        );
//    volSymmTensorField R(Rheader, mesh_);

    // Q10
    Info<< "    Calculating Q10    " << endl;
    //tmp<volTensorField> tgradU = fvc::grad(U_);
    //const volTensorField& gradU = tgradU();  //using const-ref object, not the tmp func.
    //Rall_= ((2.0/3.0)*I)*k_ - this->nut_*dev(twoSymm(tgradU())) + nonlinearStress_;
    R_ = this->R();
//  volScalarField Q10 = Foam::sqrt(Rstress_ && Rstress_)/(k_ + kSMALL + Foam::sqrt(Rstress_ && Rstress_));
//  volScalarField Q10 = Foam::sqrt(R && R)/(k_ + kSMALL + Foam::sqrt(R && R));
    volScalarField Q10 = Foam::sqrt(R_ && R_)/(k_ + kSMALL + Foam::sqrt(R_ && R_));
//    volScalarField Q10 (
//    IOobject (
//                "Q10",
//                runTime_.timeName(),
//                mesh_,
//                IOobject::NO_READ
//            ),
//            //Foam::sqrt(R && R)/(k_ + kSMALL + Foam::sqrt(R && R)),
//            //Foam::sqrt(Rstress_ && Rstress_)/(k_ + kSMALL + Foam::sqrt(Rstress_ && Rstress_)),
//            mesh_
//            //"zeroGradient"
//            //dimless,
//    );
     Info << "--> Q10    Min:" << Foam::min(Q10).value() << " Max :" << Foam::max(Q10).value() << endl;


    // Get trained turbulence model corrections
    #include "nonLinearModel.H"

    // Set additional anisotropy to zero for low k
    forAll(Ax_, idx)
    {
	    if (k_[idx] <= 1e-7)
	    {
	        Ax_[idx] = (Ax_[idx]-Ax_[idx]);
	    }
    }

    volScalarField S2(2*magSqr(symm(gradU)));
    volScalarField G(GName(), nut_*S2);
    
    volScalarField Gc = nut_*S2 - 2*k_ * (Ax_ && symm(gradU));
    volScalarField Rc = 2*k_ * (Rx_ && symm(gradU));

    dimensionedScalar min_nut
    (
        "min_nut",
        dimensionSet(0, 2, -1, 0, 0, 0 ,0),
        1e-25
    );

    // Update omega and G at the wall
    omega_.boundaryField().updateCoeffs();

    const volScalarField CDkOmega
    (
        (2*alphaOmega2_)*(fvc::grad(k_) & fvc::grad(omega_))/omega_
    );

    const volScalarField F1(this->F1(CDkOmega));

    // Turbulent frequency equation
    tmp<fvScalarMatrix> omegaEqn
    (
        fvm::ddt(omega_)
      + fvm::div(phi_, omega_)
      - fvm::laplacian(DomegaEff(F1), omega_)
     ==
        gamma(F1) * min(Gc /(nut_+min_nut), (c1_/a1_)*betaStar_*omega_*max(a1_*omega_, b1_*F23()*sqrt(S2)))
      + gamma(F1) * mag(Rc)/(nut_+min_nut)
      - fvm::Sp(beta(F1)*omega_, omega_)
      - fvm::SuSp
        (
            (F1 - scalar(1))*CDkOmega/omega_,
            omega_
        )
    );

    omegaEqn().relax();

    omegaEqn().boundaryManipulate(omega_.boundaryField());

    solve(omegaEqn);
    bound(omega_, omegaMin_);

    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(k_)
      + fvm::div(phi_, k_)
      - fvm::laplacian(DkEff(F1), k_)
     ==
        min(Gc, c1_*betaStar_*k_*omega_)
      + mag(Rc)
      - fvm::Sp(betaStar_*omega_, k_)
    );

    kEqn().relax();
    solve(kEqn);
    bound(k_, kMin_);


    // Re-calculate viscosity
    nut_ = a1_*k_/max(a1_*omega_, b1_*F23()*sqrt(S2));
    nut_.correctBoundaryConditions();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace incompressible
} // End namespace Foam

// ************************************************************************* //
