program basic_run_unpreturbed


  use init_module
  use parameters_module
  use setup_module
  use io_module
  use integrator


  implicit none

  character (len=20):: target_name
  real (kind=8) :: t0, time_interval
  integer (kind=4) :: nstep_local
  real (kind=8) :: rrr
  integer :: nparticles1, nparticles2

  integer::narg
  character *300 buffer
  character *50 shortbuf
  character *50 fBase, f1, f2 
 


!------------------------------------------------------
!
!
!

! set the disk parameters
  call RANDOM_SEED()


! set the target parameters
! set the target parameters
  nparticles1 = 5000
  nparticles2 = 4000
  call DEFAULT_PARAMETERS
!  call DEFAULT_PARAMETERS_ARG(nparticles1, nparticles2)


  call CREATE_COLLISION

!
!     -----loop over the system for the output
!

! initialize rk routine
  call INIT_RKVAR(x0, mass1, mass2, epsilon1, epsilon2, theta1, phi1, &
       theta2, phi2, rscale1, rscale2, rout1, rout2, n, n1, n2)


  t0 = tstart

  nstep = int( (tend - t0) / h) + 2
  nstep_local = nstep 

  time_interval = (tend - t0) * 2

  nunit = 50
  call OCTAVE_PARAMETERS_OUT(mass1, theta1, phi1, rout1, mass2, &
       theta2, phi2, rout2, original_rv(1:3), original_rv(4:6), time_interval, x0(n,:), n, nunit)


  narg = IARGC()
  if(narg .gt. 0) then
    call GETARG(1,buffer)
    shortbuf = TRIM(buffer)
    buffer = ''
    if(shortbuf .eq. '-o') then
      call GETARG(2,buffer)
      fBase = TRIM(buffer)
      
      f1 = "basic_unp_"//trim(fBase)//".out"
      f2 = "basic_"//trim(fBase)//".out"
      f2 = f2//".out"
    else
      f1 = "a.000"
      f2 = "a.101"
    end if
  end if
  
  print*, f1
  print*, f2


!      call CREATE_IMAGES
!  write(fname,'(a5)') 'a.000'
  open(unit, file=f1)
  call OUTPUT_PARTICLES(unit, x0, mass1, mass2, &
       eps1, eps2, &
       n, n1, n2, &
       time, header_on)
  close(unit)


! main integration loop
  iout = 0
  do istep = 1, nstep_local
    call TAKE_A_STEP
    rrr = sqrt(x0(n,1)*x0(n,1)+x0(n,2)*x0(n,2) + x0(n,3)*x0(n,3))
!    print*,istep, time, rrr
    if (mod(istep, 50) == 5) then
!      call CREATE_IMAGES
    endif
  enddo

!      call CREATE_IMAGES
!  write(fname,'(a5)') 'a.101'
  open(unit, file=f2)
  call OUTPUT_PARTICLES(unit, x0, mass1, mass2, &
       eps1, eps2, &
       n, n1, n2, &
       time, header_on)
  close(unit)

! this creates a simple script for animating the output with gnuplot
! gnuplot 
! i = 1
! j = 2
! load 'gscript
  if (.not. header_on) then
    call CREATE_GNUPLOT_SCRIPT(x0, iout)
  endif

! clean up memory
  deallocate(x0)
  deallocate(xout)
  call DEALLOCATE_RKVAR


! enddo


end program basic_run_unpreturbed


