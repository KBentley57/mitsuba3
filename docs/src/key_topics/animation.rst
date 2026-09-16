.. _sec-animation:

Animation and motion blur
=========================

Mitsuba can interpolate transformations over time and integrate over a sensor's
shutter interval to render motion blur. This page covers how to specify animated
transformations and details backend-specific behaviors.

Overview
--------

An animated transformation consists of a sequence of keyframes, each pairing a
timestamp with a transformation matrix. Each keyframe is decomposed into scale,
rotation (quaternion), and translation components. Evaluating the animation at a
given time interpolates rotations via `spherical linear interpolation (Slerp)
<https://en.wikipedia.org/wiki/Spherical_linear_interpolation>`_ and linearly
interpolates scale and translation.

In XML, an animation replaces the ``<transform>`` tag. In Python, an animated
transformation is specified by passing a :py:class:`mitsuba.AnimatedTransform4f`
(constructed from a dictionary of timestamp-transform pairs or a list of
``(time, transform)`` tuples) wherever a transform is expected.

.. tabs::
    .. code-tab:: xml

        <animation name="to_world">
            <transform time="0.0">
                <translate value="0, 0, 0"/>
            </transform>
            <transform time="1.0">
                <translate value="1, 0, 0"/>
            </transform>
        </animation>

    .. code-tab:: python

        'to_world': mi.AnimatedTransform4f({
            0.0: mi.ScalarTransform4f.translate([0, 0, 0]),
            1.0: mi.ScalarTransform4f.translate([1, 0, 0]),
        })

An ``<animation>`` can carry an ``id`` to be shared across objects via
``<ref>`` (see :ref:`sec-file-format`).

Keyframe times must be **strictly increasing**. Animations only support affine
transformations **without shear**. Multi-keyframe animations containing shear
will raise an error. Constant (single-keyframe) transformations are evaluated as
plain matrices and may contain shear.


.. _sec-animation-shutter:

The shutter interval
--------------------

To render motion blur, the sensor must specify an active shutter interval via
two parameters.

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Parameter
      - Default
      - Description
    * - ``shutter_open``
      - ``0.0``
      - Time at which the shutter opens
    * - ``shutter_close``
      - ``0.0``
      - Time at which the shutter closes

Camera rays sample their time uniformly from
:math:`[\mathtt{shutter\_open}, \mathtt{shutter\_close}]`. Because
``shutter_close`` defaults to ``0.0``, the interval is empty by default and no
motion blur is rendered unless this value is set explicitly.

The shutter interval and keyframe times share the same arbitrary time units. For
optimal rendering performance, the keyframes should roughly match the camera's
shutter interval, rather than spanning a much wider time range. Concretely, this
means using keyframes that correspond to the active frame, rather than storing
keyframes that model a longer sequence.


What can be animated
--------------------

.. list-table::
    :header-rows: 1
    :widths: 28 20 52

    * - Object
      - Animated ``to_world``
      - Notes
    * - Sensors
      - Yes
      - Keyframes may be spaced arbitrarily.
    * - ``point``, ``spot``, ``directional``, ``projector``, ``envmap``
      - Yes
      - Keyframes may be spaced arbitrarily.
    * - ``sunsky``, ``timed_sunsky``
      - No
      - Rejected with an error, see :ref:`below <sec-animation-emitters>`.
    * - Area emitters
      - No
      - Not expressible, see :ref:`below <sec-animation-emitters>`.
    * - Shapes
      - Only via ``instance``
      - Requires **evenly spaced** keyframes, see
        :ref:`below <sec-animation-shapes>`.

Only rigid and affine transformations of whole objects (scale, rotation,
translation) can be animated. Deforming geometry is not supported.

Animated sensors
----------------

The ``perspective``, ``thinlens``, ``orthographic``, ``radiancemeter``, and
``distant`` sensors all support animated ``to_world`` transformations.

.. tabs::
    .. code-tab:: xml

        <sensor type="perspective">
            <float name="shutter_open" value="0.0"/>
            <float name="shutter_close" value="1.0"/>

            <animation name="to_world">
                <transform time="0.0">
                    <lookat origin="0, 0, -5" target="0, 0, 0" up="0, 1, 0"/>
                </transform>
                <transform time="1.0">
                    <lookat origin="2, 0, -5" target="0, 0, 0" up="0, 1, 0"/>
                </transform>
            </animation>
        </sensor>

    .. code-tab:: python

        {
            'type': 'perspective',
            'shutter_open': 0.0,
            'shutter_close': 1.0,
            'to_world': mi.AnimatedTransform4f({
                0.0: mi.ScalarTransform4f.look_at(origin=[0, 0, -5], target=[0, 0, 0], up=[0, 1, 0]),
                1.0: mi.ScalarTransform4f.look_at(origin=[2, 0, -5], target=[0, 0, 0], up=[0, 1, 0]),
            }),
        }

.. image:: ../../../resources/data/docs/images/render/animation_sensor.jpg
    :width: 100%
    :align: center

In this example, the scene remains static while the camera dollies sideways
during the shutter interval, causing the entire frame to blur.

.. _sec-animation-emitters:

Animated emitters
-----------------

Emitters can also be animated using an animated ``to_world`` transformation.

.. tabs::
    .. code-tab:: xml

        <emitter type="spot">
            <animation name="to_world">
                <transform time="0.0">
                    <lookat origin="-3.7, -2.1, 3.6" target="0.3, 0.0, 0.6" up="0, 0, 1"/>
                </transform>
                <transform time="1.0">
                    <lookat origin="-0.9, -4.4, 3.6" target="0.3, 0.0, 0.6" up="0, 0, 1"/>
                </transform>
            </animation>
        </emitter>

    .. code-tab:: python

        {
            'type': 'spot',
            'to_world': mi.AnimatedTransform4f({
                0.0: mi.ScalarTransform4f.look_at(origin=[-3.7, -2.1, 3.6], target=[0.3, 0.0, 0.6], up=[0, 0, 1]),
                1.0: mi.ScalarTransform4f.look_at(origin=[-0.9, -4.4, 3.6], target=[0.3, 0.0, 0.6], up=[0, 0, 1]),
            }),
        }

.. subfigstart::
.. subfigure:: ../../../resources/data/docs/images/render/animation_emitter_t0.jpg
   :caption: Shutter closed at :math:`t = 0`
.. subfigure:: ../../../resources/data/docs/images/render/animation_emitter.jpg
   :caption: Shutter open over :math:`[0, 1]`
.. subfigure:: ../../../resources/data/docs/images/render/animation_emitter_t1.jpg
   :caption: Shutter closed at :math:`t = 1`
.. subfigend::
    :label: fig-animation-emitter

In this example, a spot light moves along an arc around a static bunny. The
geometry stays sharp while the cast shadow smears across the shutter interval.

**Limitations**

- **Area emitters cannot be animated**. Shapes are animated through instancing
  (via ``shapegroup``), which does not support attached emitters.
- **Sunsky emitters reject animated transforms**. Rotating the sky dome does not
  physically model the sun's trajectory. Use the ``timed_sunsky`` plugin
  instead, which models sun movement over time using date and time parameters.

.. _sec-animation-shapes:

Animated shapes
---------------

Animating shapes requires instancing via the :ref:`shape-instance` plugin.
Motion is evaluated directly by the active ray tracing backend (Embree, OptiX,
or Metal).

.. tabs::
    .. code-tab:: xml

        <shape type="shapegroup" id="my_group">
            <shape type="sphere"/>
        </shape>

        <shape type="instance">
            <ref id="my_group"/>
            <animation name="to_world">
                <transform time="0.0">
                    <translate value="0, 0, 0"/>
                </transform>
                <transform time="1.0">
                    <translate value="0, 0, 1"/>
                </transform>
            </animation>
        </shape>

    .. code-tab:: python

        'my_group': {
            'type': 'shapegroup',
            'shape': {'type': 'sphere'},
        },
        'my_instance': {
            'type': 'instance',
            'shape': {'type': 'ref', 'id': 'my_group'},
            'to_world': mi.AnimatedTransform4f({
                0.0: mi.ScalarTransform4f.translate([0, 0, 0]),
                1.0: mi.ScalarTransform4f.translate([0, 0, 1]),
            }),
        }

.. image:: ../../../resources/data/docs/images/render/animation_shape.jpg
    :width: 100%
    :align: center

In this example, the bunny sweeps across the frame while turning slightly, under
static camera and lighting.

.. note::

    Animated instances require **evenly spaced** keyframes. Embree, OptiX, and
    Metal define motion via a keyframe count and time interval, interpolating
    intermediate times uniformly. Keyframes with non-uniform spacing will raise
    an error. (This restriction does not apply to sensors or emitters.)

Backend differences
*******************

Ray tracing backends differ in how they implement motion blur, most notably in
how they interpolate rotations.

.. list-table::
    :header-rows: 1
    :widths: 20 20 60

    * - Backend
      - Rotation
      - Animated instances
    * - Embree (``llvm_*``, ``scalar_*``)
      - Slerp
      - Fully supported and matches Mitsuba's internal evaluation. At most 129
        keyframes per instance. Out-of-range instances **disappear** (see :ref:`below <sec-animation-time-range>`).
    * - OptiX (``cuda_*``)
      - Nlerp
      - Fully supported. At most 65535 keyframes per instance. Out-of-range
        instances are **clamped**.
    * - Metal (``metal_*``)
      - Nlerp
      - Fully supported and matches OptiX. Out-of-range instances are **clamped**.
    * - Native kd-tree
      - --
      - Instancing and animated shapes are not supported.

Slerp (spherical linear interpolation) interpolates rotations at constant angular
speed. Nlerp (normalized linear interpolation) linearly blends quaternion
components and normalizes the result.

The teapot below rotates 150 degrees about the vertical axis between two keyframes, rendered with a closed
shutter (``shutter_open == shutter_close``) to capture instantaneous poses
(:monosp:`resources/data/docs/scenes/animation_rotation.xml`).

.. subfigstart::
.. subfigure:: ../../../resources/data/docs/images/render/animation_rotation_embree.jpg
   :caption: Embree, slerp (:math:`-40.5^\circ`)
.. subfigure:: ../../../resources/data/docs/images/render/animation_rotation_optix.jpg
   :caption: OptiX, nlerp (:math:`-45.0^\circ`)
.. subfigure:: ../../../resources/data/docs/images/render/animation_rotation_metal.jpg
   :caption: Metal, nlerp (:math:`-45.0^\circ`)
.. subfigend::
    :label: fig-animation-rotation

Slerp and Nlerp agree exactly at :math:`t = 0`, :math:`t = 0.5`, and
:math:`t = 1`. For this rotation, their disagreement peaks near
:math:`t \approx 0.23` and :math:`t \approx 0.77`, shown above.

Opening the shutter across the full interval accumulates these pose differences
into motion blur.

.. subfigstart::
.. subfigure:: ../../../resources/data/docs/images/render/animation_rotation_blur_embree.jpg
   :caption: Embree, slerp
.. subfigure:: ../../../resources/data/docs/images/render/animation_rotation_blur_optix.jpg
   :caption: OptiX, nlerp
.. subfigure:: ../../../resources/data/docs/images/render/animation_rotation_blur_metal.jpg
   :caption: Metal, nlerp
.. subfigend::
    :label: fig-animation-rotation-blur

Both methods follow the same rotational arc, so blurred silhouettes have the
same extent but differ in density, as visible in the teapot's spout.

For instances with large rotations between keyframes, GPU renders may subtly
deviate from CPU renders due to this non-constant angular velocity. The angular
error stays under 0.1 degrees for rotations up to roughly 43 degrees, but increases
to about 0.9 degrees at 90 degrees and 4.5 degrees at 150 degrees. Subdividing
large rotations with additional keyframes reduces this difference.

.. _sec-animation-time-range:

Instances with differing time ranges
************************************

Instances need not share a common time range. The scene-wide range is their
union. However, when a ray's time falls *outside* an instance's specified
keyframe range, backend behaviors differ.

- **OptiX and Metal** clamp the time, keeping the instance frozen at its first or
  last keyframe pose.
- **Embree** hides the instance outside its keyframe range and emits a warning
  at scene load time.

To achieve identical rendering across all backends, ensure every animated
instance has keyframes spanning the entire camera shutter interval (e.g., by
duplicating boundary poses on the uniform keyframe grid).

Animation and ``mitsuba.traverse()``
------------------------------------

Through :py:func:`mitsuba.traverse()`, an animated transformation exposes its
keyframes as four component tensors.

.. list-table::
    :header-rows: 1
    :widths: 22 18 60

    * - Parameter
      - Shape
      - Contents
    * - ``times``
      - ``(N,)``
      - Keyframe times, strictly increasing
    * - ``scale``
      - ``(N, 3)``
      - Per-axis scale factors
    * - ``rotation``
      - ``(N, 4)``
      - Rotation quaternions, ``(x, y, z, w)``
    * - ``translation``
      - ``(N, 3)``
      - Translations

Editing one component leaves the others untouched.

.. code-block:: python

    params = mi.traverse(scene)
    t = mi.TensorXf(params['instance.to_world.translation'])
    t[1, 0] = 2.5                                   # second keyframe, x axis
    params['instance.to_world.translation'] = t
    params.update()

The four tensors share a packed buffer and must have matching keyframe counts.
Call :py:func:`mitsuba.traverse()` again after changing the keyframe count.

When a transformation contains only a **single keyframe**, it additionally
exposes a plain 4x4 matrix directly under its parent name (e.g.,
``params['sensor.to_world']``) for backwards compatibility. That matrix takes
precedence if written alongside the component views.

.. code-block:: python

    params['sensor.to_world'] = mi.ScalarTransform4f.translate([0, 0, 1])
    params.update()

.. note::

    Animated instances are not currently differentiable. Backend acceleration
    structures rebuild instances from host-side representations, so gradients
    do not propagate to instance keyframe components.


