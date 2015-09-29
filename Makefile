cabal_files=rbm.cabal

hs_files=Data/RBM.hs\
			Data/MLP.hs\
			Data/Matrix.hs\
			Data/DNN/Trainer.hs\
			Examples/Mnist.hs

tix_files=perf-RBM.tix\
			 trainbatches.tix\
			 testbatches.tix\
			 mnist.tix\
			 test-RBM.tix\
			 bigtrainbatches.tix\
			 generatetrainlabels.tix\
			 console.tix

#all:mnist
all:dist/cabal.test.ok dist/cabal.build.ok

dist/cabal.test.ok:$(hs_files) dist/setup-config tix
	cabal test 2>&1
	@touch $@

dist/cabal.build.ok:$(hs_files) dist/setup-config tix
	cabal build 2>&1
	@touch $@

clean:tix
	cabal clean

test:tix
	cabal test 2>&1

mnist:tix
	cabal build mnist
	./dist/build/mnist/mnist +RTS -N

batches:tix
	cabal build testbatches
	cabal build trainbatches
	rm dist/test* || echo ok
	rm dist/train* || echo ok
	./dist/build/trainbatches/trainbatches
	rm -f $(tix_files)
	./dist/build/testbatches/testbatches
	rm -f $(tix_files)

tix:
	rm -f $(tix_files)

dist/setup-config:$(cabal_files) Makefile
	cabal install --only-dependencies
	cabal configure --enable-coverage --enable-tests
	@touch $@

DATA=dist/train-images-idx3-ubyte.gz \
	  dist/train-labels-idx1-ubyte.gz \
	  dist/t10k-images-idx3-ubyte.gz \
	  dist/t10k-labels-idx1-ubyte.gz

data:$(DATA)

dist/train-images-idx3-ubyte.gz:
	mkdir -p $(@D)
	cd dist && wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

dist/train-labels-idx1-ubyte.gz:
	mkdir -p $(@D)
	cd dist && wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

dist/t10k-images-idx3-ubyte.gz:
	mkdir -p $(@D)
	cd dist && wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz

dist/t10k-labels-idx1-ubyte.gz:
	mkdir -p $(@D)
	cd dist && wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

$$%:;@$(call true)$(info $(call or,$$$*))
